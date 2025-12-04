#!/usr/bin/env python3
"""Test single frame with v4 processing to compare quality."""

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


def test_frame(video_path, frame_num=100, output_prefix='test_v4'):
    """Extract and process a single frame for quality comparison."""
    
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
    print(f"Saved original frame: {output_prefix}_original.jpg")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
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
    base_size = 672  # High quality
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
    
    # Save raw matte
    matte_vis = (matte_raw * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_raw.jpg', matte_vis)
    print(f"Saved raw matte: {output_prefix}_matte_raw.jpg")
    
    # Process with v4 pipeline
    matte_np = matte_raw.copy()
    
    # Step 1: Sharpen - более агрессивно
    matte_np = sharpen_alpha_aggressive(matte_np, threshold=0.5, steepness=25)
    matte_vis = (matte_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_sharpened.jpg', matte_vis)
    
    # Step 2: Hard threshold
    matte_np = hard_threshold_mask(matte_np, 0.12, 0.88)
    matte_vis = (matte_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_threshold.jpg', matte_vis)
    
    # Step 3: Morphology - мягкий erode
    matte_np = apply_morphological_cleanup(matte_np, erode_size=2, dilate_size=2)
    matte_vis = (matte_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_morph.jpg', matte_vis)
    
    # Step 4: Remove small components
    matte_np = remove_small_components(matte_np, min_area_ratio=0.01)
    matte_vis = (matte_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_clean.jpg', matte_vis)
    
    # Step 5: Guided filter
    matte_np = guided_filter_fast(frame_resized, matte_np, radius=3, eps=0.001)
    matte_np = np.clip(matte_np, 0, 1)
    
    # Step 5.5: Detect and clean remaining halo (light version)
    bg_color = (128, 128, 128)
    matte_np = detect_and_clean_halo(frame_resized, matte_np, bg_color, threshold=30)
    
    # Step 5.6: Remove bright artifacts near mask edges (ear, etc.)
    frame_resized, matte_np = remove_bright_artifacts(
        frame_resized, matte_np, bg_color,
        edge_width=10, brightness_threshold=35, color_threshold=45
    )
    
    matte_vis = (matte_np * 255).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_matte_final.jpg', matte_vis)
    
    print(f"Saved final matte: {output_prefix}_matte_final.jpg")
    
    # Step 6: Defringe - удаление цветного ореола
    frame_defringed = remove_fringe(frame_resized, matte_np, fringe_width=3)
    
    # Step 7: Enhance quality
    frame_enhanced = enhance_output_quality(
        frame_defringed, matte_np,
        sharpen_amount=0.3,
        contrast_boost=1.05,
        saturation_boost=1.1,
        denoise=True
    )
    
    # Composite result
    matte_3d = np.stack([matte_np] * 3, axis=-1)
    bg = np.full(frame_resized.shape, bg_color, dtype=np.float32)
    result = matte_3d * frame_enhanced.astype(np.float32) + (1 - matte_3d) * bg
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.resize(result, (w, h), cv2.INTER_LANCZOS4)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f'{output_prefix}_result.jpg', result_bgr)
    print(f"Saved result: {output_prefix}_result.jpg")
    
    print("\n✅ Done! Check the output files to compare quality.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', default='squat.mp4')
    parser.add_argument('--frame', '-f', type=int, default=100)
    parser.add_argument('--output', '-o', default='test_v4')
    args = parser.parse_args()
    
    test_frame(args.video, args.frame, args.output)
