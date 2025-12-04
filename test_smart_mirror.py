#!/usr/bin/env python3
"""
Test Professional Relighting Pipeline
Подход: Intrinsic Decomposition → New Lighting → MODNet

Ключевое: сохраняем идентичность (без diffusion)!
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

def load_intrinsic_model():
    """Load Intrinsic Image Decomposition model"""
    sys.path.insert(0, '/Users/user/Documents/pers2/Intrinsic')
    
    from intrinsic.pipeline import run_pipeline, load_models
    
    # Use CPU/MPS for MacBook
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    models = load_models('paper_weights', device=device)
    return models, run_pipeline

def decompose_image(image_bgr: np.ndarray, models, run_pipeline):
    """
    Decompose image into albedo and shading
    Returns RGB float images [0-1]
    """
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_norm = rgb.astype(np.float32) / 255.0
    
    result = run_pipeline(
        models,
        rgb_norm,
        resize_conf=0.0,
        linear=False,
        device=device
    )
    
    return result['albedo'], result['shading']

def create_studio_shading(original_shading: np.ndarray, preset: str = "soft") -> np.ndarray:
    """
    Create new professional studio lighting
    
    Presets:
    - soft: Мягкий заполняющий свет (минимум теней)
    - dramatic: Контрастный боковой свет
    - top: Верхний свет как в спортзале
    - flat: Полностью плоский свет
    """
    h, w = original_shading.shape[:2]
    
    # Convert to grayscale if RGB
    if len(original_shading.shape) == 3:
        shading_gray = np.mean(original_shading, axis=2)
    else:
        shading_gray = original_shading.copy()
    
    if preset == "soft":
        # Мягкий студийный свет - уменьшаем влияние теней
        # Сохраняем структуру но поднимаем тени
        base = 0.7  # Минимальный уровень света
        new_shading = shading_gray * (1 - base) + base
        
        # Лёгкий градиент сверху
        gradient = np.linspace(1.05, 0.98, h).reshape(-1, 1)
        new_shading = new_shading * gradient
        
    elif preset == "dramatic":
        # Драматичный свет слева
        x_grad = np.linspace(1.2, 0.85, w)
        y_grad = np.linspace(1.05, 0.95, h).reshape(-1, 1)
        light_map = np.outer(y_grad.flatten(), x_grad)
        
        base = 0.5
        new_shading = shading_gray * (1 - base) + base
        new_shading = new_shading * light_map
        
    elif preset == "top":
        # Верхний свет (как в спортзале)
        y_grad = np.linspace(1.15, 0.85, h).reshape(-1, 1)
        light_map = np.tile(y_grad, (1, w))
        
        base = 0.6
        new_shading = shading_gray * (1 - base) + base
        new_shading = new_shading * light_map
        
    elif preset == "flat":
        # Полностью плоский свет
        new_shading = np.ones_like(shading_gray) * 0.9
        
    else:
        new_shading = shading_gray
    
    # Clamp
    new_shading = np.clip(new_shading, 0.3, 1.2)
    
    # Convert to RGB
    return np.stack([new_shading] * 3, axis=-1).astype(np.float32)

def recompose(albedo: np.ndarray, shading: np.ndarray) -> np.ndarray:
    """Recompose image from albedo and shading"""
    result = albedo * shading
    result = np.clip(result, 0, 1)
    
    # Slight gamma for better look
    result = np.power(result, 0.97)
    
    return (result * 255).astype(np.uint8)

def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """Remove background using MODNet, return RGBA"""
    import torch
    import torch.nn.functional as F
    
    sys.path.insert(0, '/Users/user/Documents/pers2/MODNet')
    from src.models.modnet import MODNet
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    ckpt = '/Users/user/Documents/pers2/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    modnet.load_state_dict(torch.load(ckpt, map_location='cpu'))
    modnet = modnet.module.to(device)
    modnet.eval()
    
    h, w = image_bgr.shape[:2]
    
    # Resize to multiple of 32
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (new_w, new_h))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, matte = modnet(img_t, True)
    
    matte = F.interpolate(matte, size=(h, w), mode='bilinear', align_corners=False)
    matte = matte[0, 0].cpu().numpy()
    matte = (matte * 255).astype(np.uint8)
    
    # Clean matte
    matte = cv2.GaussianBlur(matte, (3, 3), 0)
    
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = matte
    
    return rgba

def add_background(rgba: np.ndarray, bg_type: str = "gray") -> np.ndarray:
    """Add professional background"""
    h, w = rgba.shape[:2]
    
    if bg_type == "gray":
        bg = np.full((h, w, 3), 145, dtype=np.uint8)
    elif bg_type == "gradient":
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            val = int(170 - (i / h) * 50)
            bg[i, :] = [val, val, val]
    elif bg_type == "dark":
        bg = np.full((h, w, 3), 80, dtype=np.uint8)
    else:
        bg = np.full((h, w, 3), 128, dtype=np.uint8)
    
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    bg_f = bg.astype(np.float32)
    
    result = fg * alpha + bg_f * (1 - alpha)
    return result.astype(np.uint8)

def apply_color_grade(image: np.ndarray, preset: str = "warm") -> np.ndarray:
    """Apply color grading for professional look"""
    img = image.astype(np.float32) / 255.0
    
    if preset == "warm":
        # Warm tones for healthy skin
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.04, 0, 1)  # R up
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.96, 0, 1)  # B down
        img = np.clip((img - 0.5) * 1.05 + 0.52, 0, 1)  # Contrast + lift
        
    elif preset == "cool":
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.03, 0, 1)  # B up
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.97, 0, 1)  # R down
        
    elif preset == "vibrant":
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    return (img * 255).astype(np.uint8)

def process_professional(
    input_path: str,
    output_dir: str = "output",
    lighting: str = "soft",
    background: str = "gradient",
    color: str = "warm"
):
    """
    Full professional processing pipeline
    """
    print(f"\n🎬 Professional Relighting (Identity-Preserving)")
    print(f"=" * 50)
    print(f"Input: {input_path}")
    print(f"Lighting: {lighting}")
    print(f"Background: {background}")
    print(f"Color: {color}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Cannot read {input_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Size: {w}x{h}")
    
    os.makedirs(output_dir, exist_ok=True)
    base = Path(input_path).stem
    
    # Step 1: Intrinsic decomposition
    print("\n1️⃣ Intrinsic Decomposition...")
    try:
        models, run_pipeline = load_intrinsic_model()
        albedo, shading = decompose_image(image, models, run_pipeline)
        print(f"   ✅ Albedo shape: {albedo.shape}")
        print(f"   ✅ Shading shape: {shading.shape}")
        
        # Save albedo and shading
        cv2.imwrite(f"{output_dir}/{base}_1_albedo.png", 
                   cv2.cvtColor((albedo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{base}_2_shading_orig.png",
                   cv2.cvtColor((shading * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                   
    except Exception as e:
        print(f"   ❌ Intrinsic failed: {e}")
        print("   Using simple brightness adjustment instead...")
        albedo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        shading = np.ones_like(albedo) * 0.8
    
    # Step 2: Create new lighting
    print("\n2️⃣ Creating Studio Lighting...")
    new_shading = create_studio_shading(shading, lighting)
    print(f"   ✅ Preset: {lighting}")
    
    cv2.imwrite(f"{output_dir}/{base}_3_shading_new.png",
               (new_shading * 255).astype(np.uint8))
    
    # Step 3: Recompose
    print("\n3️⃣ Recomposing Image...")
    relit = recompose(albedo, new_shading)
    relit_bgr = cv2.cvtColor(relit, cv2.COLOR_RGB2BGR)
    print("   ✅ Done")
    
    cv2.imwrite(f"{output_dir}/{base}_4_relit.png", relit_bgr)
    
    # Step 4: Remove background
    print("\n4️⃣ Removing Background (MODNet)...")
    try:
        rgba = remove_background(relit_bgr)
        print("   ✅ Done")
        
        cv2.imwrite(f"{output_dir}/{base}_5_cutout.png", rgba)
    except Exception as e:
        print(f"   ❌ MODNet failed: {e}")
        rgba = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
    
    # Step 5: Add background
    print("\n5️⃣ Adding Background...")
    result = add_background(rgba, background)
    print(f"   ✅ Type: {background}")
    
    # Step 6: Color grade
    print("\n6️⃣ Color Grading...")
    result = apply_color_grade(result, color)
    print(f"   ✅ Preset: {color}")
    
    # Save final
    final_path = f"{output_dir}/{base}_FINAL.png"
    cv2.imwrite(final_path, result)
    
    print(f"\n✅ DONE! Saved to: {final_path}")
    print(f"\n📁 All outputs in {output_dir}/:")
    print(f"   - {base}_1_albedo.png")
    print(f"   - {base}_2_shading_orig.png")
    print(f"   - {base}_3_shading_new.png")
    print(f"   - {base}_4_relit.png")
    print(f"   - {base}_5_cutout.png")
    print(f"   - {base}_FINAL.png")
    
    return final_path


def main():
    if len(sys.argv) < 2:
        print("""
Professional Relighting Pipeline (Identity-Preserving)
======================================================

Этот pipeline НЕ использует diffusion models, поэтому
сохраняет идентичность человека на 100%!

Usage:
  python test_smart_mirror.py <image> [lighting] [background] [color]

Lighting presets:
  - soft (default): Мягкий заполняющий свет
  - dramatic: Контрастный боковой свет
  - top: Верхний свет как в спортзале
  - flat: Полностью плоский свет

Background types:
  - gradient (default): Градиент серого
  - gray: Однотонный серый
  - dark: Тёмный фон

Color grades:
  - warm (default): Тёплые тона для кожи
  - cool: Холодные тона
  - vibrant: Насыщенные цвета

Example:
  python test_smart_mirror.py trainer_frame.png soft gradient warm
""")
        return
    
    input_path = sys.argv[1]
    lighting = sys.argv[2] if len(sys.argv) > 2 else "soft"
    background = sys.argv[3] if len(sys.argv) > 3 else "gradient"
    color = sys.argv[4] if len(sys.argv) > 4 else "warm"
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    process_professional(input_path, "output", lighting, background, color)


if __name__ == "__main__":
    main()
