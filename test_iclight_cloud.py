#!/usr/bin/env python3
"""
Тест IC-Light Cloud API.

1. Извлекает кадр из видео
2. Удаляет фон через MODNet
3. Отправляет на IC-Light для relighting
4. Сохраняет результаты
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Добавляем пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet/src'))

import torch
import torch.nn.functional as F
from models.modnet import MODNet


def load_modnet():
    """Загружаем MODNet модель."""
    model_path = 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    
    state_dict = torch.load(model_path, map_location='cpu')
    modnet.load_state_dict(state_dict)
    modnet = modnet.module
    modnet.to(device)
    modnet.eval()
    
    return modnet, device


def get_mask(modnet, device, frame):
    """Получаем маску через MODNet."""
    h, w = frame.shape[:2]
    
    # Подготовка изображения
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    tensor = tensor.to(device)
    
    with torch.no_grad():
        _, _, matte = modnet(tensor, True)
    
    matte = matte.squeeze().cpu().numpy()
    matte = cv2.resize(matte, (w, h))
    
    return matte


def extract_person_with_alpha(frame, mask):
    """Создаёт изображение с альфа-каналом."""
    # Normalize mask
    mask = np.clip(mask, 0, 1)
    
    # Убираем шумы
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, mask_clean = cv2.threshold(mask_uint8, 30, 255, cv2.THRESH_BINARY)
    mask_clean = mask_clean.astype(np.float32) / 255.0
    
    # Небольшое сглаживание
    mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    
    # RGB + Alpha
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgba = np.dstack([img_rgb, (mask_clean * 255).astype(np.uint8)])
    
    return Image.fromarray(img_rgba, 'RGBA')


def main():
    print("="*60)
    print("IC-Light Cloud Test")
    print("="*60)
    
    # 1. Загружаем MODNet
    print("\n[1/4] Loading MODNet...")
    modnet, device = load_modnet()
    
    # 2. Извлекаем кадр из видео
    print("\n[2/4] Extracting frame from video...")
    video_path = "squat.mp4"
    if not os.path.exists(video_path):
        # Ищем любое видео
        for f in os.listdir('.'):
            if f.endswith(('.mp4', '.mov', '.avi')):
                video_path = f
                break
    
    if not os.path.exists(video_path):
        print("❌ No video found! Please provide squat.mp4 or similar")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    # Берём кадр из середины
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read frame")
        sys.exit(1)
    
    print(f"   Frame size: {frame.shape}")
    
    # 3. Получаем маску и вырезаем человека
    print("\n[3/4] Removing background with MODNet...")
    mask = get_mask(modnet, device, frame)
    
    # Создаём PNG с прозрачным фоном
    person_rgba = extract_person_with_alpha(frame, mask)
    fg_path = "iclight_input.png"
    person_rgba.save(fg_path)
    print(f"   Saved: {fg_path}")
    
    # 4. Отправляем на IC-Light
    print("\n[4/4] Sending to IC-Light Cloud...")
    from iclight_client import ICLightClient
    
    client = ICLightClient()
    
    # Тестируем разные пресеты
    presets_to_test = ['studio', 'dramatic', 'neon']
    
    for preset in presets_to_test:
        print(f"\n--- Testing preset: {preset} ---")
        try:
            results = client.relight_with_preset(
                fg_path, 
                preset=preset,
                lighting='left'  # Свет слева
            )
            
            if results:
                output_path = f"iclight_{preset}.png"
                results[0].save(output_path)
                print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Error with {preset}: {e}")
    
    print("\n" + "="*60)
    print("Done! Check the output files:")
    print("  - iclight_input.png (input with transparent bg)")
    print("  - iclight_studio.png")
    print("  - iclight_dramatic.png")
    print("  - iclight_neon.png")
    print("="*60)


if __name__ == '__main__':
    main()
