#!/usr/bin/env python3
"""
Правильный подход к relighting:
Простая цветокоррекция - работает предсказуемо и не ломает изображение.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet/src'))

def load_modnet():
    import torch
    from models.modnet import MODNet
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    state_dict = torch.load('MODNet/pretrained/modnet_webcam_portrait_matting.ckpt', map_location='cpu')
    modnet.load_state_dict(state_dict)
    modnet = modnet.module.to(device)
    modnet.eval()
    return modnet, device

def get_mask(modnet, device, frame):
    import torch
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        _, _, matte = modnet(tensor, True)
    matte = matte.squeeze().cpu().numpy()
    return cv2.resize(matte, (w, h))

def simple_studio_relight(image_rgb, style='bright'):
    img = image_rgb.astype(np.float32) / 255.0
    
    if style == 'bright':
        img = np.power(img, 0.85)
        img = img * 1.1 + 0.05
    elif style == 'warm':
        img = np.power(img, 0.9)
        img[:,:,0] *= 1.08
        img[:,:,1] *= 1.02
        img[:,:,2] *= 0.92
    elif style == 'cool':
        img = np.power(img, 0.9)
        img[:,:,0] *= 0.92
        img[:,:,1] *= 0.98
        img[:,:,2] *= 1.08
    elif style == 'high_contrast':
        img = np.power(img, 0.85)
        img = (img - 0.5) * 1.2 + 0.5
        img = img * 1.05
    elif style == 'soft':
        img = np.power(img, 0.95)
        bright = cv2.GaussianBlur(img, (0, 0), 30)
        img = img * 0.85 + bright * 0.15
        img = img * 1.1
    elif style == 'fitness_pro':
        img = np.power(img, 0.88)
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] *= 1.15
        hsv[:,:,2] *= 1.05
        hsv = np.clip(hsv, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        img[:,:,0] *= 1.03
        img[:,:,2] *= 0.97
    
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def apply_quality_cutout(image_rgb, mask, bg_color=(128, 128, 128)):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_uint8 = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
    _, mask_clean = cv2.threshold(mask_uint8, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    mask_smooth = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    alpha = mask_smooth.astype(np.float32) / 255.0
    
    bg = np.full_like(image_rgb, bg_color, dtype=np.float32)
    fg = image_rgb.astype(np.float32)
    result = fg * alpha[:,:,np.newaxis] + bg * (1 - alpha[:,:,np.newaxis])
    return result.astype(np.uint8)

def main():
    print("="*60)
    print("Quality Relighting - Simple & Reliable")
    print("="*60)
    
    cap = cv2.VideoCapture('squat.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("Loading MODNet...")
    modnet, device = load_modnet()
    mask = get_mask(modnet, device, frame)
    
    styles = ['bright', 'warm', 'cool', 'high_contrast', 'soft', 'fitness_pro']
    
    for style in styles:
        relit = simple_studio_relight(frame_rgb, style)
        result = apply_quality_cutout(relit, mask)
        cv2.imwrite(f"quality_{style}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Saved: quality_{style}.png")
    
    original = apply_quality_cutout(frame_rgb, mask)
    cv2.imwrite("quality_original.png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    print("Saved: quality_original.png")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
