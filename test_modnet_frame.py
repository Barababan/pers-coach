import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Add MODNet src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet'))
from src.models.modnet import MODNet


def test_modnet_frame(video_path, frame_num=250, output_path='test_modnet_output.jpg'):
    """
    Test MODNet on a single frame from video.
    
    Args:
        video_path: Path to input video
        frame_num: Frame number to extract
        output_path: Path to save result
    """
    print(f"Testing MODNet on frame {frame_num} from {video_path}")
    
    # Check MPS availability
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
        print("Please download from: https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view")
        return
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if device.type == 'cpu':
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    else:
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        modnet = modnet.to(device)
    
    modnet.eval()
    print("✅ Model loaded successfully")
    
    # Load video and extract frame
    print(f"\nExtracting frame {frame_num}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"❌ Failed to read frame {frame_num}")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Prepare frame for MODNet
    # Resize to multiple of 32
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (rw, rh), cv2.INTER_AREA)
    
    # Transform
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    frame_pil = Image.fromarray(frame_resized)
    frame_tensor = torch_transforms(frame_pil)
    frame_tensor = frame_tensor[None, :, :, :]
    
    if device.type != 'cpu':
        frame_tensor = frame_tensor.to(device)
    
    # Inference
    print("\nRunning inference...")
    start_time = time.time()
    
    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    
    inference_time = time.time() - start_time
    print(f"✅ Inference time: {inference_time:.3f}s")
    
    # Process matte
    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    
    # Create gray background (128, 128, 128)
    bg_color = 128
    result_np = matte_np * frame_resized + (1 - matte_np) * np.full(frame_resized.shape, bg_color)
    result_np = result_np.astype(np.uint8)
    
    # Resize back to original size
    result_np = cv2.resize(result_np, (w, h))
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    
    # Save result
    cv2.imwrite(output_path, result_bgr)
    print(f"\n✅ Saved result to: {output_path}")
    
    # Performance estimate
    total_frames = 5793  # squat.mp4
    estimated_time = total_frames * inference_time
    print(f"\n📊 Performance estimate for full video ({total_frames} frames):")
    print(f"   Time per frame: {inference_time:.3f}s")
    print(f"   Total time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
    
    return inference_time


if __name__ == "__main__":
    if not os.path.exists("squat.mp4"):
        print("❌ squat.mp4 not found in current directory")
        sys.exit(1)
    
    test_modnet_frame("squat.mp4", frame_num=250)
