#!/usr/bin/env python3
"""
Test AI enhancement on single frame.
Tests: GFPGAN, Real-ESRGAN, and color grading.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet'))
from src.models.modnet import MODNet

from process_video_modnet_v4 import (
    sharpen_alpha_aggressive,
    hard_threshold_mask, 
    apply_morphological_cleanup,
    remove_small_components,
    guided_filter_fast,
    remove_fringe,
    detect_and_clean_halo,
    remove_bright_artifacts,
    enhance_output_quality
)

from process_video_modnet_v5 import (
    GFPGANEnhancer,
    RealESRGANEnhancer,
    apply_neural_color_correction
)


def test_ai_enhancement(video_path, frame_num=100, output_prefix='test_ai'):
    """Test AI enhancement on a single frame."""
    
    print(f"Loading frame {frame_num} from {video_path}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Failed to read frame {frame_num}")
        return
    
    # Save original
    cv2.imwrite(f'{output_prefix}_original.jpg', frame)
    print(f"✅ Saved: {output_prefix}_original.jpg")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load MODNet
    print("\nLoading MODNet...")
    pretrained_ckpt = 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    if device.type != 'cpu':
        modnet = modnet.to(device)
    modnet.eval()
    
    # Prepare frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Processing size
    base_size = 672
    if w >= h:
        rh = base_size
        rw = int(w / h * base_size)
    else:
        rw = base_size
        rh = int(h / w * base_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    frame_resized = cv2.resize(frame_rgb, (rw, rh), cv2.INTER_AREA)
    
    # Transform
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    frame_pil = Image.fromarray(frame_resized)
    frame_tensor = torch_transforms(frame_pil)[None, :, :, :]
    if device.type != 'cpu':
        frame_tensor = frame_tensor.to(device)
    
    # MODNet inference
    print("Running MODNet inference...")
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    matte_raw = matte_tensor[0, 0].data.cpu().numpy()
    
    # Process mask (v4 pipeline)
    print("Processing mask...")
    matte_np = matte_raw.copy()
    matte_np = sharpen_alpha_aggressive(matte_np, threshold=0.5, steepness=22)
    matte_np = hard_threshold_mask(matte_np, 0.15, 0.85)
    matte_np = apply_morphological_cleanup(matte_np, erode_size=2, dilate_size=2)
    matte_np = remove_small_components(matte_np, min_area_ratio=0.01)
    matte_np = guided_filter_fast(frame_resized, matte_np, radius=3, eps=0.001)
    matte_np = np.clip(matte_np, 0, 1)
    
    bg_color = (128, 128, 128)
    matte_np = detect_and_clean_halo(frame_resized, matte_np, bg_color, threshold=30)
    frame_resized, matte_np = remove_bright_artifacts(
        frame_resized, matte_np, bg_color,
        edge_width=10, brightness_threshold=35, color_threshold=45
    )
    
    # Defringe and enhance
    frame_processed = remove_fringe(frame_resized, matte_np, fringe_width=3)
    frame_processed = enhance_output_quality(
        frame_processed, matte_np,
        sharpen_amount=0.3,
        contrast_boost=1.05,
        saturation_boost=1.1,
        denoise=True
    )
    
    # Composite
    matte_3d = np.stack([matte_np] * 3, axis=-1)
    bg = np.full(frame_resized.shape, bg_color, dtype=np.float32)
    result = matte_3d * frame_processed.astype(np.float32) + (1 - matte_3d) * bg
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.resize(result, (w, h), cv2.INTER_LANCZOS4)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Save base result
    cv2.imwrite(f'{output_prefix}_base.jpg', result_bgr)
    print(f"✅ Saved: {output_prefix}_base.jpg")
    
    # === Test Color Grading ===
    print("\n--- Testing Color Grading ---")
    color_modes = ['warm', 'cool', 'vivid', 'cinematic', 'natural']
    
    for mode in color_modes:
        colored = apply_neural_color_correction(result_bgr.copy(), mode=mode)
        cv2.imwrite(f'{output_prefix}_color_{mode}.jpg', colored)
        print(f"✅ Saved: {output_prefix}_color_{mode}.jpg")
    
    # === Test GFPGAN ===
    print("\n--- Testing GFPGAN ---")
    gfpgan = GFPGANEnhancer()
    
    if gfpgan.enabled:
        start = time.time()
        gfpgan_result = gfpgan.enhance(result_bgr.copy())
        elapsed = time.time() - start
        cv2.imwrite(f'{output_prefix}_gfpgan.jpg', gfpgan_result)
        print(f"✅ Saved: {output_prefix}_gfpgan.jpg ({elapsed:.2f}s)")
        
        # GFPGAN + color grading
        gfpgan_warm = apply_neural_color_correction(gfpgan_result, mode='warm')
        cv2.imwrite(f'{output_prefix}_gfpgan_warm.jpg', gfpgan_warm)
        print(f"✅ Saved: {output_prefix}_gfpgan_warm.jpg")
    else:
        print("⚠️  GFPGAN not available")
    
    # === Test Real-ESRGAN ===
    print("\n--- Testing Real-ESRGAN ---")
    
    # Test x2 model
    realesrgan_x2 = RealESRGANEnhancer(model_name='RealESRGAN_x2plus', scale=2)
    
    if realesrgan_x2.enabled:
        # Test on smaller crop to speed up
        crop_h, crop_w = 256, 256
        y_start = h // 2 - crop_h // 2
        x_start = w // 2 - crop_w // 2
        crop = result_bgr[y_start:y_start+crop_h, x_start:x_start+crop_w].copy()
        
        cv2.imwrite(f'{output_prefix}_crop_original.jpg', crop)
        
        start = time.time()
        crop_enhanced = realesrgan_x2.enhance(crop)
        elapsed = time.time() - start
        
        cv2.imwrite(f'{output_prefix}_crop_esrgan_x2.jpg', crop_enhanced)
        print(f"✅ Saved: {output_prefix}_crop_esrgan_x2.jpg ({elapsed:.2f}s)")
        print(f"   Original: {crop.shape}, Enhanced: {crop_enhanced.shape}")
    else:
        print("⚠️  Real-ESRGAN not available")
    
    print("\n✅ All tests complete!")
    print(f"\nOutput files:")
    print(f"  Base:        {output_prefix}_base.jpg")
    print(f"  Color:       {output_prefix}_color_*.jpg")
    print(f"  GFPGAN:      {output_prefix}_gfpgan*.jpg")
    print(f"  Real-ESRGAN: {output_prefix}_crop_*.jpg")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', default='squat.mp4')
    parser.add_argument('--frame', '-f', type=int, default=100)
    parser.add_argument('--output', '-o', default='test_ai')
    args = parser.parse_args()
    
    test_ai_enhancement(args.video, args.frame, args.output)
