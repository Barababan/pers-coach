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


def process_video_modnet(input_path, output_path, bg_color=(128, 128, 128)):
    """
    Process video with MODNet background removal and gray background.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        bg_color: Background color as RGB tuple (default: gray 128,128,128)
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"Background color: RGB{bg_color}")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders) - M1 GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (slower)")
    
    # Load MODNet model
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
    
    print(f"\nVideo info: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    # Calculate resize dimensions (multiple of 32)
    if width >= height:
        rh = 512
        rw = int(width / height * 512)
    else:
        rw = 512
        rh = int(height / width * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    print(f"Processing size: {rw}x{rh}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transform
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
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
            
            # Process matte
            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            
            # Apply gray background
            result_np = matte_np * frame_resized + (1 - matte_np) * np.full(frame_resized.shape, bg_color[0])
            result_np = result_np.astype(np.uint8)
            
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
    parser = argparse.ArgumentParser(description='Process video with MODNet background removal')
    parser.add_argument('--input', '-i', type=str, default='squat.mp4', help='Input video path')
    parser.add_argument('--output', '-o', type=str, default='squat_gray_modnet.mp4', help='Output video path')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[128, 128, 128], 
                        help='Background color RGB (default: 128 128 128)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    process_video_modnet(args.input, args.output, tuple(args.bg_color))
