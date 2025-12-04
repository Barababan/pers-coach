#!/usr/bin/env python3
"""
Test professional lighting on a processed frame with background removed.
Shows all lighting styles side by side.
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

from professional_lighting import (
    LightingStyle,
    professional_lighting_pipeline,
    normalize_illumination,
    apply_virtual_studio_light,
    enhance_skin_tones
)


def process_frame_with_modnet(frame_bgr, modnet, device):
    """Process frame through MODNet pipeline."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    
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
    
    # Inference
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    matte_raw = matte_tensor[0, 0].data.cpu().numpy()
    
    # Process mask
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
        sharpen_amount=0.3, contrast_boost=1.05, saturation_boost=1.1, denoise=True
    )
    
    # Composite
    matte_3d = np.stack([matte_np] * 3, axis=-1)
    bg = np.full(frame_resized.shape, bg_color, dtype=np.float32)
    result = matte_3d * frame_processed.astype(np.float32) + (1 - matte_3d) * bg
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.resize(result, (w, h), cv2.INTER_LANCZOS4)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Resize mask to original size
    mask_full = cv2.resize(matte_np, (w, h))
    
    return result_bgr, mask_full


def test_lighting_styles(video_path, frame_num=100, output_prefix='test_pro_lighting'):
    """Test all professional lighting styles on a frame."""
    
    print(f"Loading frame {frame_num} from {video_path}")
    
    # Load video
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
    
    # Process frame
    print("Processing frame with MODNet...")
    base_result, mask = process_frame_with_modnet(frame, modnet, device)
    
    # Save base result (no lighting)
    cv2.imwrite(f'{output_prefix}_base.jpg', base_result)
    print(f"✅ Saved: {output_prefix}_base.jpg")
    
    # Test each lighting style
    print("\nTesting lighting styles...")
    
    styles_to_test = [
        (LightingStyle.NEUTRAL, "neutral"),
        (LightingStyle.WARM_STUDIO, "warm_studio"),
        (LightingStyle.COOL_STUDIO, "cool_studio"),
        (LightingStyle.FITNESS, "fitness"),
        (LightingStyle.BROADCAST, "broadcast"),
        (LightingStyle.CINEMATIC, "cinematic"),
        (LightingStyle.DAYLIGHT, "daylight"),
    ]
    
    for style, name in styles_to_test:
        result = professional_lighting_pipeline(
            base_result.copy(),
            mask=mask,
            style=style,
            normalize_light=True,
            enhance_skin=True,
            virtual_studio=False,
            adaptive_exposure_enabled=True
        )
        cv2.imwrite(f'{output_prefix}_{name}.jpg', result)
        print(f"✅ Saved: {output_prefix}_{name}.jpg")
    
    # Test with virtual studio lighting
    print("\nTesting with virtual studio light...")
    for style, name in [(LightingStyle.FITNESS, "fitness"), (LightingStyle.BROADCAST, "broadcast")]:
        result = professional_lighting_pipeline(
            base_result.copy(),
            mask=mask,
            style=style,
            normalize_light=True,
            enhance_skin=True,
            virtual_studio=True,
            adaptive_exposure_enabled=True
        )
        cv2.imwrite(f'{output_prefix}_{name}_studio.jpg', result)
        print(f"✅ Saved: {output_prefix}_{name}_studio.jpg")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    images = []
    labels = ['Base', 'Neutral', 'Warm Studio', 'Cool Studio', 'Fitness', 'Broadcast', 'Cinematic', 'Daylight']
    
    for i, label in enumerate(labels):
        if label == 'Base':
            img = cv2.imread(f'{output_prefix}_base.jpg')
        else:
            name = label.lower().replace(' ', '_')
            img = cv2.imread(f'{output_prefix}_{name}.jpg')
        
        if img is not None:
            # Add label
            h, w = img.shape[:2]
            labeled = img.copy()
            cv2.rectangle(labeled, (0, h-30), (w, h), (0, 0, 0), -1)
            cv2.putText(labeled, label, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            images.append(labeled)
    
    # Create 2x4 grid
    if len(images) >= 8:
        row1 = np.hstack(images[:4])
        row2 = np.hstack(images[4:8])
        grid = np.vstack([row1, row2])
        
        # Resize if too large
        max_width = 1920
        if grid.shape[1] > max_width:
            scale = max_width / grid.shape[1]
            new_h = int(grid.shape[0] * scale)
            grid = cv2.resize(grid, (max_width, new_h))
        
        cv2.imwrite(f'{output_prefix}_comparison.jpg', grid)
        print(f"✅ Saved: {output_prefix}_comparison.jpg")
    
    print("\n✅ All tests complete!")
    print(f"\nRecommended for fitness videos: --lighting fitness --normalize-light --enhance-skin")
    print(f"For TV-style: --lighting broadcast --normalize-light --enhance-skin")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', default='squat.mp4')
    parser.add_argument('--frame', '-f', type=int, default=100)
    parser.add_argument('--output', '-o', default='test_pro_lighting')
    args = parser.parse_args()
    
    test_lighting_styles(args.video, args.frame, args.output)
