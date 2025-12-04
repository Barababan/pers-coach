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


def sharpen_alpha_matte(matte, threshold=0.5, edge_width=0.1):
    """
    Sharpen alpha matte to reduce semi-transparent edges.
    
    Args:
        matte: Alpha matte (0-1)
        threshold: Center point for sigmoid (0.5 = middle)
        edge_width: Controls sharpness (smaller = sharper)
    """
    # Apply sigmoid to sharpen
    # This pushes values closer to 0 or 1, reducing semi-transparency
    sharpened = 1 / (1 + np.exp(-(matte - threshold) / edge_width))
    return sharpened


def apply_morphological_refinement(mask, kernel_size=3):
    """Apply light morphological operations to clean up mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def guided_filter_alpha(image, alpha, radius=8, eps=1e-6):
    """
    Apply guided filter to alpha matte using image as guidance.
    Preserves edges while smoothing.
    """
    # Convert to float
    image_f = image.astype(np.float32) / 255.0
    alpha_f = alpha.astype(np.float32)
    
    # Use OpenCV's ximgproc guided filter if available
    try:
        import cv2.ximgproc as ximgproc
        filtered = ximgproc.guidedFilter(
            guide=image_f,
            src=alpha_f,
            radius=radius,
            eps=eps
        )
        return filtered
    except:
        # Fallback to simple bilateral filter
        return cv2.bilateralFilter(alpha_f, d=radius, sigmaColor=0.1, sigmaSpace=radius)


def temporal_smooth_adaptive(current_mask, previous_smoothed, alpha_base=0.5, motion_threshold=0.1):
    """
    Adaptive temporal smoothing based on mask difference.
    Less smoothing where there's more change (motion).
    """
    if previous_smoothed is None:
        return current_mask
    
    # Calculate difference (motion indicator)
    diff = np.abs(current_mask - previous_smoothed)
    
    # Adaptive alpha: less smoothing where diff is high
    # Where diff > threshold, use less smoothing (higher alpha)
    alpha_adaptive = np.where(diff > motion_threshold, 
                               alpha_base + 0.3,  # More current frame
                               alpha_base - 0.1)  # More smoothing
    alpha_adaptive = np.clip(alpha_adaptive, 0.3, 0.9)
    
    return alpha_adaptive * current_mask + (1 - alpha_adaptive) * previous_smoothed


def process_video_modnet_v3(
    input_path, 
    output_path, 
    bg_color=(128, 128, 128),
    temporal_smooth=0.5,
    sharpen_edges=True,
    use_guided_filter=True,
    resolution_scale=1.0
):
    """
    Process video with MODNet + improved edge handling (v3).
    
    Improvements over v2:
    - Sharper alpha matte (less semi-transparency)
    - Guided filter instead of bilateral (better edge preservation)
    - Adaptive temporal smoothing (less blur in motion)
    - Lighter morphological operations
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"v3 Improvements: sharpen_edges={sharpen_edges}, guided_filter={use_guided_filter}")
    
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
            
            # Get matte as numpy (single channel, 0-1)
            matte_np = matte_tensor[0, 0].data.cpu().numpy()
            
            # 1. Light morphological cleanup (reduced from v2)
            matte_np = apply_morphological_refinement(matte_np, kernel_size=3)
            
            # 2. Sharpen alpha to reduce semi-transparency
            if sharpen_edges:
                matte_np = sharpen_alpha_matte(matte_np, threshold=0.5, edge_width=0.08)
            
            # 3. Guided filter for edge-preserving smoothing
            if use_guided_filter:
                matte_np = guided_filter_alpha(frame_resized, matte_np, radius=4, eps=1e-6)
            
            # 4. Adaptive temporal smoothing
            if temporal_smooth > 0:
                matte_np = temporal_smooth_adaptive(
                    matte_np, 
                    prev_smoothed_mask, 
                    alpha_base=temporal_smooth,
                    motion_threshold=0.15
                )
                prev_smoothed_mask = matte_np.copy()
            
            # Ensure values are in [0, 1]
            matte_np = np.clip(matte_np, 0, 1)
            
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
    parser = argparse.ArgumentParser(description='Process video with MODNet v3 (improved edges)')
    parser.add_argument('--input', '-i', type=str, default='squat.mp4')
    parser.add_argument('--output', '-o', type=str, default='squat_gray_modnet_v3.mp4')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--temporal-smooth', type=float, default=0.5, 
                        help='Temporal smoothing (0=off, 0.5=moderate, 0.8=heavy)')
    parser.add_argument('--no-sharpen', action='store_true', help='Disable edge sharpening')
    parser.add_argument('--no-guided-filter', action='store_true', help='Disable guided filter')
    parser.add_argument('--resolution', type=float, default=1.0)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    process_video_modnet_v3(
        args.input, 
        args.output, 
        tuple(args.bg_color),
        temporal_smooth=args.temporal_smooth,
        sharpen_edges=not args.no_sharpen,
        use_guided_filter=not args.no_guided_filter,
        resolution_scale=args.resolution
    )
