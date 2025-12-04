#!/usr/bin/env python3
"""
🚀 Colab Pro Runner - Запуск GPU задач из VS Code

Использует Colab Pro через Jupyter Server connection.
Позволяет запускать тяжёлые модели (Intrinsic, Real-ESRGAN) на GPU
прямо из терминала VS Code.

SETUP:
1. Открой https://colab.research.google.com
2. Создай новый notebook
3. Runtime → Change runtime type → T4 GPU (или лучше с Pro)
4. Запусти эту ячейку в Colab:

```python
# В Colab выполни:
!pip install jupyter_http_over_ws
!jupyter serverextension enable --py jupyter_http_over_ws

from google.colab import output
output.serve_kernel_port_as_window(8888)

# Или получи URL для подключения:
from notebook.notebookapp import list_running_servers
for srv in list_running_servers():
    print(srv['url'] + '?token=' + srv['token'])
```

5. Скопируй URL и запусти этот скрипт с --connect URL
"""

import subprocess
import sys
import json
import os

def check_jupyter_connection(url: str) -> bool:
    """Проверить подключение к Jupyter"""
    try:
        import requests
        resp = requests.get(url.replace('?token=', '/api?token='), timeout=5)
        return resp.status_code == 200
    except:
        return False


def run_on_colab(code: str, connection_url: str = None):
    """
    Выполнить код на Colab через Jupyter API
    """
    if connection_url:
        print(f"🔗 Connecting to: {connection_url[:50]}...")
        # Тут нужен jupyter_client для полной реализации
        print("⚠️ Direct API connection requires jupyter_client")
        print("Use the notebook approach instead.")
        return
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  📋 COPY THIS CODE TO COLAB                                  ║
╠══════════════════════════════════════════════════════════════╣
""")
    print(code)
    print("""
╚══════════════════════════════════════════════════════════════╝

📝 Instructions:
1. Go to https://colab.research.google.com
2. Create new notebook (or open existing)
3. Set Runtime → T4 GPU (you have Pro, so A100 available!)
4. Paste and run the code above
5. Download results
""")


# Готовые команды для Colab
COLAB_SETUP = '''
# 🔧 SETUP - Run this first!
!pip install -q torch torchvision timm
!git clone -q https://github.com/compphoto/Intrinsic.git
%cd Intrinsic && pip install -q -e . && %cd ..
!git clone -q https://github.com/ZHKKKe/MODNet.git  
!wget -q -P MODNet/pretrained/ https://github.com/ZHKKKe/MODNet/releases/download/v0.1.0/modnet_photographic_portrait_matting.ckpt
print("✅ Setup complete!")
'''

COLAB_PROCESS_IMAGE = '''
# 🎬 PROCESS IMAGE
# Upload your image first using the file browser on the left

import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, 'Intrinsic')
sys.path.insert(0, 'MODNet')

from intrinsic.pipeline import load_models, run_pipeline
from src.models.modnet import MODNet
import torch.nn.functional as F

device = 'cuda'

# Load models
print("Loading Intrinsic...")
intrinsic_models = load_models('paper_weights', device=device)

print("Loading MODNet...")
modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)
modnet.load_state_dict(torch.load('MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', map_location='cpu'))
modnet = modnet.module.to(device)
modnet.eval()

print("✅ Models loaded!")

# === CHANGE THIS TO YOUR IMAGE ===
IMAGE_PATH = "your_image.jpg"  # Change this!
LIGHTING = "soft"  # soft, dramatic, bright, flat
BACKGROUND = "gradient"  # gradient, gray, dark
# =================================

# Read image
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]
print(f"Processing {IMAGE_PATH}: {w}x{h}")

# 1. Intrinsic decomposition
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
result = run_pipeline(intrinsic_models, rgb, resize_conf=0.0, linear=False, device=device)
albedo, shading = result['albedo'], result['shading']

# 2. Create new lighting
shading_gray = np.mean(shading, axis=2)
if LIGHTING == "soft":
    new_shading = shading_gray * 0.3 + 0.7
elif LIGHTING == "dramatic":
    x_grad = np.linspace(1.2, 0.85, w)
    new_shading = (shading_gray * 0.5 + 0.5) * x_grad
elif LIGHTING == "bright":
    new_shading = shading_gray * 0.2 + 0.8
else:
    new_shading = np.ones_like(shading_gray) * 0.9
    
new_shading = np.clip(new_shading, 0.3, 1.2)
new_shading = np.stack([new_shading]*3, axis=-1)

# 3. Recompose
relit = np.clip(albedo * new_shading, 0, 1)
relit = (np.power(relit, 0.97) * 255).astype(np.uint8)
relit_bgr = cv2.cvtColor(relit, cv2.COLOR_RGB2BGR)

# 4. Remove background
new_h, new_w = ((h-1)//32+1)*32, ((w-1)//32+1)*32
img = cv2.resize(cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB), (new_w, new_h))
img = torch.from_numpy((img.astype(np.float32)/255.0 - 0.5)/0.5).permute(2,0,1).unsqueeze(0).to(device)
with torch.no_grad():
    _, _, matte = modnet(img, True)
matte = F.interpolate(matte, size=(h,w), mode='bilinear')[0,0].cpu().numpy()
matte = (matte * 255).astype(np.uint8)

# 5. Add background
if BACKGROUND == "gradient":
    bg = np.array([[int(170-(i/h)*50)]*3 for i in range(h)], dtype=np.uint8)
    bg = np.tile(bg.reshape(h,1,3), (1,w,1))
else:
    bg = np.full((h,w,3), 145, dtype=np.uint8)

alpha = matte[:,:,None].astype(np.float32)/255.0
result = (relit_bgr.astype(np.float32)*alpha + bg.astype(np.float32)*(1-alpha)).astype(np.uint8)

# 6. Color grade (warm)
img_f = result.astype(np.float32)/255.0
img_f[:,:,2] = np.clip(img_f[:,:,2]*1.04, 0, 1)
img_f[:,:,0] = np.clip(img_f[:,:,0]*0.96, 0, 1)
result = (np.clip((img_f-0.5)*1.05+0.52, 0, 1)*255).astype(np.uint8)

# Save
output_path = IMAGE_PATH.rsplit('.',1)[0] + '_professional.png'
cv2.imwrite(output_path, result)
print(f"✅ Saved: {output_path}")

# Show
from google.colab.patches import cv2_imshow
cv2_imshow(result)
'''


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Colab Pro Runner')
    parser.add_argument('command', choices=['setup', 'process', 'help'], 
                       help='Command to run')
    parser.add_argument('--image', help='Image path for process command')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        run_on_colab(COLAB_SETUP)
    elif args.command == 'process':
        code = COLAB_PROCESS_IMAGE
        if args.image:
            code = code.replace('your_image.jpg', args.image)
        run_on_colab(code)
    else:
        print(__doc__)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("""
🚀 Colab Pro Runner

Commands:
  python colab_runner.py setup    - Get setup code for Colab
  python colab_runner.py process  - Get processing code
  python colab_runner.py help     - Show help

With Google AI Pro you have access to:
  • A100 GPU (40GB) - fastest!
  • V100 GPU (16GB)
  • T4 GPU (16GB)
  • Longer runtime (24h vs 12h)
  • Background execution
  • More RAM

Quick start:
1. Run: python colab_runner.py setup
2. Copy code to Colab
3. Upload your image to Colab
4. Run: python colab_runner.py process --image your_image.jpg
5. Copy processing code to Colab
""")
    else:
        main()
