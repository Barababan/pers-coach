#!/usr/bin/env python3
"""
Test Real-ESRGAN locally on MacBook
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

def upscale_local(image_path: str, output_path: str = None, scale: int = 2):
    """
    Upscale image using Real-ESRGAN locally
    """
    print(f"🔍 Real-ESRGAN (local): Processing {image_path}")
    
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # Initialize model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=scale)
        
        # Model path - will download if not exists
        model_name = f'RealESRGAN_x{scale}plus'
        model_path = f'weights/{model_name}.pth'
        
        if not os.path.exists(model_path):
            print(f"  Downloading model {model_name}...")
            os.makedirs('weights', exist_ok=True)
            import urllib.request
            url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
            urllib.request.urlretrieve(url, model_path)
            print(f"  ✅ Downloaded to {model_path}")
        
        # Create upsampler
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,  # No tiling for small images
            tile_pad=10,
            pre_pad=0,
            half=False  # Full precision for M1
        )
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ Cannot read {image_path}")
            return None
        
        print(f"  Input: {img.shape[1]}x{img.shape[0]}")
        
        # Upscale
        output, _ = upsampler.enhance(img, outscale=scale)
        
        print(f"  Output: {output.shape[1]}x{output.shape[0]}")
        
        # Save
        if output_path is None:
            base = Path(image_path).stem
            output_path = f"output/{base}_x{scale}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)
        
        print(f"✅ Saved to {output_path}")
        return output_path
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying alternative method...")
        return upscale_opencv(image_path, output_path, scale)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def upscale_opencv(image_path: str, output_path: str = None, scale: int = 2):
    """
    Fallback: Simple OpenCV upscaling with enhancement
    """
    print(f"🔍 OpenCV Upscale: Processing {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read {image_path}")
        return None
    
    h, w = img.shape[:2]
    print(f"  Input: {w}x{h}")
    
    # Upscale using INTER_LANCZOS4 (best quality)
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # Blend original upscaled with sharpened
    result = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
    
    # Denoise slightly
    result = cv2.bilateralFilter(result, 5, 50, 50)
    
    print(f"  Output: {result.shape[1]}x{result.shape[0]}")
    
    if output_path is None:
        base = Path(image_path).stem
        output_path = f"output/{base}_opencv_x{scale}.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    
    print(f"✅ Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_local_upscale.py <image_path> [scale]")
        print("Example: python test_local_upscale.py frame.png 2")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)
    
    result = upscale_local(image_path, scale=scale)
