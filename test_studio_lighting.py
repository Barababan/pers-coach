#!/usr/bin/env python3
"""
Test Studio Lighting - тестирует профессиональное студийное освещение 
с MODNet сегментацией на видео.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet'))
from src.models.modnet import MODNet

from studio_lighting import (
    apply_studio_lighting, 
    LightingPreset,
    visualize_lighting_setup
)


def get_modnet_mask(frame_bgr, modnet, device):
    """Получает маску от MODNet."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    
    # Processing size
    base_size = 512
    if w >= h:
        rh = base_size
        rw = int(w / h * base_size)
    else:
        rw = base_size
        rh = int(h / w * base_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    frame_resized = cv2.resize(frame_rgb, (rw, rh), cv2.INTER_AREA)
    
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    frame_pil = Image.fromarray(frame_resized)
    frame_tensor = torch_transforms(frame_pil)[None, :, :, :]
    if device.type != 'cpu':
        frame_tensor = frame_tensor.to(device)
    
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    
    matte = matte_tensor[0, 0].data.cpu().numpy()
    
    # Resize to original
    matte = cv2.resize(matte, (w, h))
    
    # Бинаризация для чистых краёв
    matte = np.where(matte > 0.5, 1.0, 0.0).astype(np.float32)
    
    # Небольшое сглаживание краёв
    matte = cv2.GaussianBlur(matte, (5, 5), 0)
    
    return matte


def apply_lighting_and_composite(image_bgr, mask, preset, bg_color=(128, 128, 128)):
    """
    Применяет освещение и композитит на фон.
    
    ВАЖНО: Освещение применяется ДО вырезания!
    """
    h, w = image_bgr.shape[:2]
    
    # 1. Применяем студийное освещение к ПОЛНОМУ изображению
    lit_image = apply_studio_lighting(
        image_bgr, 
        mask, 
        preset=preset,
        ambient=0.2,
        strength=0.7
    )
    
    # 2. Композитим на фон
    mask_3d = mask[:, :, np.newaxis]
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    result = (mask_3d * lit_image.astype(np.float32) + 
              (1 - mask_3d) * bg.astype(np.float32))
    
    return np.clip(result, 0, 255).astype(np.uint8)


def test_studio_lighting(video_path, frame_num=250, output_prefix='studio'):
    """Тестирует студийное освещение."""
    
    print(f"Loading frame {frame_num} from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Failed to read frame {frame_num}")
        return
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load MODNet
    print("Loading MODNet...")
    pretrained_ckpt = 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    if device.type != 'cpu':
        modnet = modnet.to(device)
    modnet.eval()
    
    # Get mask
    print("Getting mask...")
    mask = get_modnet_mask(frame, modnet, device)
    
    # Save original
    cv2.imwrite(f'{output_prefix}_original.jpg', frame)
    print(f"✅ Original: {output_prefix}_original.jpg")
    
    # Save mask
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_mask.jpg', mask_vis)
    print(f"✅ Mask: {output_prefix}_mask.jpg")
    
    # Test all lighting presets
    print("\nTesting lighting presets...")
    results = []
    
    presets = [
        LightingPreset.NATURAL,
        LightingPreset.FITNESS_STUDIO,
        LightingPreset.CYBERPUNK,
        LightingPreset.NEON_GYM,
        LightingPreset.SUNSET_WARM,
        LightingPreset.COOL_PROFESSIONAL,
        LightingPreset.DRAMATIC,
        LightingPreset.RGB_TRICOLOR,
    ]
    
    for preset in presets:
        result = apply_lighting_and_composite(frame, mask, preset)
        
        output_path = f'{output_prefix}_{preset.value}.jpg'
        cv2.imwrite(output_path, result)
        print(f"✅ {preset.value}: {output_path}")
        
        # Resize for comparison
        result_small = cv2.resize(result, (270, 480))
        
        # Add label
        cv2.rectangle(result_small, (0, 450), (270, 480), (0, 0, 0), -1)
        cv2.putText(result_small, preset.value, (5, 472),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        results.append(result_small)
    
    # Create comparison grid (2 rows x 4 cols)
    print("\nCreating comparison...")
    row1 = np.hstack(results[:4])
    row2 = np.hstack(results[4:])
    comparison = np.vstack([row1, row2])
    
    cv2.imwrite(f'{output_prefix}_comparison.jpg', comparison)
    print(f"✅ Comparison: {output_prefix}_comparison.jpg")
    
    # Also save lighting setup diagrams
    setups = []
    for preset in presets:
        setup = visualize_lighting_setup(preset, size=270)
        setups.append(setup)
    
    setup_row1 = np.hstack(setups[:4])
    setup_row2 = np.hstack(setups[4:])
    setup_comparison = np.vstack([setup_row1, setup_row2])
    
    cv2.imwrite(f'{output_prefix}_setups.jpg', setup_comparison)
    print(f"✅ Lighting setups: {output_prefix}_setups.jpg")
    
    print("\n✅ Studio lighting test complete!")
    print("\nРекомендации:")
    print("  - 'cyberpunk' - модный синий/фиолетовый свет")
    print("  - 'neon_gym' - неоновый розовый/бирюзовый")
    print("  - 'fitness_studio' - классический студийный")
    print("  - 'dramatic' - контрастный для эффектных кадров")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', default='squat.mp4')
    parser.add_argument('--frame', '-f', type=int, default=250)
    parser.add_argument('--output', '-o', default='studio')
    args = parser.parse_args()
    
    test_studio_lighting(args.video, args.frame, args.output)
