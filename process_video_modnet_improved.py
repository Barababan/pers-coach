import os
import sys
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Add MODNet src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet'))
from src.models.modnet import MODNet


def apply_morphological_refinement(mask, kernel_size=5):
    """Apply morphological operations to clean up mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_bilateral_filter(mask, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to smooth while preserving edges."""
    return cv2.bilateralFilter(mask.astype(np.float32), d, sigma_color, sigma_space)


def temporal_smooth_ema(current_mask, previous_smoothed, alpha=0.7):
    """Apply exponential moving average for temporal smoothing."""
    if previous_smoothed is None:
        return current_mask
    return alpha * current_mask + (1 - alpha) * previous_smoothed


def process_video_modnet_improved(
    input_path, 
    output_path, 
    bg_color=(128, 128, 128),
    temporal_smooth=0.7,
    edge_refine=True,
    resolution_scale=1.0
):
    """
    Process video with MODNet + quality improvements.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        bg_color: Background color RGB
        temporal_smooth: EMA alpha (0=no smoothing, 0.9=heavy smoothing)
        edge_refine: Apply edge refinement
        resolution_scale: Scale processing resolution (1.0=512, 1.5=768, 2.0=1024)
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"Improvements: temporal_smooth={temporal_smooth}, edge_refine={edge_refine}, resolution_scale={resolution_scale}")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    
    # Load MODNet
    print("\nLoading MODNet model...")
    pretrained_ckpt = 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    
    if not os.path.exists(pretrained_ckpt):
        print(f"❌ Model not found: {pretrained_ckpt}")
        return
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if device.type == 'cpu':
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    else:
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        modnet = modnet.to(device)
    
    modnet.eval()
    print("✅ Model loaded")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    # Calculate processing resolution
    base_size = int(512 * resolution_scale)
    if width >= height:
        rh = base_size
        rw = int(width / height * base_size)
    else:
        rw = base_size
        rh = int(height / width * base_size)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    print(f"Processing size: {rw}x{rh} (scale={resolution_scale})")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transform
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Temporal smoothing state
    prev_smoothed_mask = None
    
    # Process frames
    print("\nProcessing frames...")
    frame_count = 0
    total_inference_time = 0
    start_time = time.time()
    
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Prepare frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (rw, rh), cv2.INTER_AREA)
            
            frame_pil = Image.fromarray(frame_resized)
            frame_tensor = torch_transforms(frame_pil)
            frame_tensor = frame_tensor[None, :, :, :]
            
            if device.type != 'cpu':
                frame_tensor = frame_tensor.to(device)
            
            # Inference
            inference_start = time.time()
            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)
            total_inference_time += time.time() - inference_start
            
            # Get matte as numpy
            matte_np = matte_tensor[0, 0].data.cpu().numpy()  # Single channel
            
            # Edge refinement
            if edge_refine:
                # Morphological operations
                matte_np = apply_morphological_refinement(matte_np, kernel_size=5)
                # Bilateral filter for smooth edges
                matte_np = apply_bilateral_filter(matte_np, d=7, sigma_color=50, sigma_space=50)
            
            # Temporal smoothing
            if temporal_smooth > 0:
                matte_np = temporal_smooth_ema(matte_np, prev_smoothed_mask, alpha=temporal_smooth)
                prev_smoothed_mask = matte_np.copy()
            
            # Expand to 3 channels for blending
            matte_3d = np.stack([matte_np] * 3, axis=-1)
            
            # Apply gray background
            result_np = matte_3d * frame_resized + (1 - matte_3d) * np.full(frame_resized.shape, bg_color[0])
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            
            # Resize back and write
            result_np = cv2.resize(result_np, (width, height))
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            out.write(result_bgr)
            
            # Update progress
            if frame_count % 10 == 0:
                avg_time = total_inference_time / frame_count
                remaining = total_frames - frame_count
                eta_seconds = remaining * avg_time
                pbar.set_postfix({
                    'avg': f'{avg_time:.2f}s/f',
                    'ETA': f'{eta_seconds/60:.1f}min'
                })
            
            pbar.update(1)
    
    cap.release()
    out.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_inference = total_inference_time / frame_count
    
    print(f"\n✅ Processing complete!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average inference: {avg_inference:.3f}s per frame")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with improved MODNet')
    parser.add_argument('--input', '-i', type=str, default='squat.mp4', help='Input video')
    parser.add_argument('--output', '-o', type=str, default='squat_gray_modnet_v2.mp4', help='Output video')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[128, 128, 128], help='Background RGB')
    parser.add_argument('--temporal-smooth', type=float, default=0.7, 
                        help='Temporal smoothing alpha (0=off, 0.9=heavy)')
    parser.add_argument('--no-edge-refine', action='store_true', help='Disable edge refinement')
    parser.add_argument('--resolution', type=float, default=1.0, 
                        help='Resolution scale (1.0=512, 1.5=768, 2.0=1024)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    process_video_modnet_improved(
        args.input, 
        args.output, 
        tuple(args.bg_color),
        temporal_smooth=args.temporal_smooth,
        edge_refine=not args.no_edge_refine,
        resolution_scale=args.resolution
    )
