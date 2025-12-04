#!/usr/bin/env python3
"""
Test Real-ESRGAN via HuggingFace Spaces (FREE)
"""

from gradio_client import Client, handle_file
import sys
import os
from pathlib import Path

def upscale_real_esrgan(image_path: str, output_path: str = None):
    """
    Upscale image using Real-ESRGAN via HuggingFace Space
    FREE but slower (queued)
    """
    print(f"🔍 Real-ESRGAN (HuggingFace): Processing {image_path}")
    
    # Try different Real-ESRGAN spaces
    spaces = [
        "doevent/Face-Real-ESRGAN",
        "akhaliq/Real-ESRGAN",
        "Nick088/Real-ESRGAN_Pytorch",
    ]
    
    for space in spaces:
        try:
            print(f"  Trying {space}...")
            client = Client(space)
            
            result = client.predict(
                handle_file(image_path),
                "RealESRGAN_x2plus",  # model
                api_name="/predict"
            )
            
            if output_path is None:
                base = Path(image_path).stem
                output_path = f"output/{base}_upscaled.png"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Result is path to output file
            import shutil
            shutil.copy(result, output_path)
            
            print(f"✅ Saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  ❌ {space} failed: {e}")
            continue
    
    print("❌ All spaces failed")
    return None


def upscale_swin2sr(image_path: str, output_path: str = None):
    """
    Alternative: Swin2SR upscaler
    """
    print(f"🔍 Swin2SR: Processing {image_path}")
    
    try:
        client = Client("Fabrice-TIERCELIN/Swin2SR")
        
        result = client.predict(
            handle_file(image_path),
            api_name="/predict"
        )
        
        if output_path is None:
            base = Path(image_path).stem
            output_path = f"output/{base}_swin2sr.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        import shutil
        shutil.copy(result, output_path)
        
        print(f"✅ Saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Swin2SR failed: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hf_upscale.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)
    
    result = upscale_real_esrgan(image_path, output_path)
    
    if result is None:
        print("\n🔄 Trying alternative Swin2SR...")
        result = upscale_swin2sr(image_path, output_path)
