#!/usr/bin/env python3
"""
Тест Intrinsic Image Decomposition для relighting с сохранением идентичности.

Принцип работы:
1. Разделяет изображение на albedo (текстура) и shading (освещение)
2. Albedo - это "кто" на изображении (сохраняется)
3. Shading - это "как освещено" (можно менять)
4. Новое изображение = albedo × новый shading
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_intrinsic():
    print("="*60)
    print("Intrinsic Image Decomposition Test")
    print("="*60)
    
    # Импортируем intrinsic
    print("\n[1/5] Importing intrinsic...")
    try:
        from chrislib.data_util import load_image, np_to_pil
        from intrinsic.pipeline import load_models, run_pipeline
        print("✅ Intrinsic imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Загружаем модели
    print("\n[2/5] Loading models (will download on first run)...")
    import torch
    # Принудительно используем CPU чтобы избежать проблем с MPS
    device = torch.device('cpu')
    models = load_models('v2', device=device)
    print("✅ Models loaded")
    
    # Загружаем тестовое изображение
    print("\n[3/5] Loading test image...")
    input_path = "iclight_input.png"  # Наше изображение с удалённым фоном
    
    if not os.path.exists(input_path):
        # Альтернатива - извлечём кадр из видео
        print("   No input found, extracting from video...")
        cap = cv2.VideoCapture("squat.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite("test_frame.png", frame)
            input_path = "test_frame.png"
        else:
            print("❌ No test image available")
            return
    
    # Intrinsic требует RGB float [0-1]
    image = load_image(input_path)
    print(f"   Image shape: {image.shape}")
    
    # Убираем альфа-канал если есть
    if image.shape[-1] == 4:
        print("   Converting RGBA to RGB...")
        # Композитируем на белом фоне
        alpha = image[:, :, 3:4]
        rgb = image[:, :, :3]
        white_bg = np.ones_like(rgb)
        image = rgb * alpha + white_bg * (1 - alpha)
        print(f"   New shape: {image.shape}")
    
    # Запускаем декомпозицию
    print("\n[4/5] Running intrinsic decomposition...")
    results = run_pipeline(models, image, device='cpu')
    
    # Компоненты
    albedo = results['hr_alb']           # Высокое разрешение albedo
    diffuse_shading = results['dif_shd'] # Диффузное освещение
    residual = results['residual']       # Остаток (блики, отражения)
    
    print(f"   Albedo shape: {albedo.shape}")
    print(f"   Shading shape: {diffuse_shading.shape}")
    
    # Сохраняем компоненты
    print("\n[5/5] Saving decomposition...")
    
    # Albedo
    albedo_pil = np_to_pil(albedo)
    albedo_pil.save("intrinsic_albedo.png")
    print("   Saved: intrinsic_albedo.png")
    
    # Shading
    shading_pil = np_to_pil(diffuse_shading)
    shading_pil.save("intrinsic_shading.png")
    print("   Saved: intrinsic_shading.png")
    
    # Residual
    if residual is not None:
        residual_pil = np_to_pil(np.clip(residual, 0, 1))
        residual_pil.save("intrinsic_residual.png")
        print("   Saved: intrinsic_residual.png")
    
    # Теперь делаем RELIGHTING!
    print("\n" + "="*60)
    print("Creating relit variants...")
    print("="*60)
    
    # Вариант 1: Более яркий свет (умножаем shading)
    brighter_shading = np.clip(diffuse_shading * 1.5, 0, 1)
    brighter = albedo * brighter_shading
    if residual is not None:
        brighter = brighter + residual
    brighter = np.clip(brighter, 0, 1)
    np_to_pil(brighter).save("intrinsic_relit_bright.png")
    print("   Saved: intrinsic_relit_bright.png")
    
    # Вариант 2: Контрастный свет
    contrast_shading = np.clip((diffuse_shading - 0.5) * 1.5 + 0.5, 0, 1)
    contrast = albedo * contrast_shading
    if residual is not None:
        contrast = contrast + residual * 0.5
    contrast = np.clip(contrast, 0, 1)
    np_to_pil(contrast).save("intrinsic_relit_contrast.png")
    print("   Saved: intrinsic_relit_contrast.png")
    
    # Вариант 3: Свет слева (градиент)
    h, w = diffuse_shading.shape[:2]
    left_gradient = np.linspace(1.2, 0.8, w).reshape(1, w, 1)
    left_shading = np.clip(diffuse_shading * left_gradient, 0, 1)
    left_lit = albedo * left_shading
    if residual is not None:
        left_lit = left_lit + residual * 0.5
    left_lit = np.clip(left_lit, 0, 1)
    np_to_pil(left_lit).save("intrinsic_relit_left.png")
    print("   Saved: intrinsic_relit_left.png")
    
    # Вариант 4: Свет сверху
    top_gradient = np.linspace(1.3, 0.7, h).reshape(h, 1, 1)
    top_shading = np.clip(diffuse_shading * top_gradient, 0, 1)
    top_lit = albedo * top_shading
    if residual is not None:
        top_lit = top_lit + residual * 0.5
    top_lit = np.clip(top_lit, 0, 1)
    np_to_pil(top_lit).save("intrinsic_relit_top.png")
    print("   Saved: intrinsic_relit_top.png")
    
    # Вариант 5: Теплый свет (добавляем жёлтый оттенок в shading)
    warm_shading = diffuse_shading.copy()
    if len(warm_shading.shape) == 2:
        warm_shading = np.stack([warm_shading]*3, axis=-1)
    warm_shading[:,:,0] *= 1.1  # Red
    warm_shading[:,:,1] *= 1.05  # Green
    warm_shading[:,:,2] *= 0.9   # Blue
    warm_shading = np.clip(warm_shading, 0, 1)
    warm_lit = albedo * warm_shading
    if residual is not None:
        warm_lit = warm_lit + residual * 0.5
    warm_lit = np.clip(warm_lit, 0, 1)
    np_to_pil(warm_lit).save("intrinsic_relit_warm.png")
    print("   Saved: intrinsic_relit_warm.png")
    
    # Вариант 6: Холодный свет
    cool_shading = diffuse_shading.copy()
    if len(cool_shading.shape) == 2:
        cool_shading = np.stack([cool_shading]*3, axis=-1)
    cool_shading[:,:,0] *= 0.9   # Red
    cool_shading[:,:,1] *= 0.95  # Green
    cool_shading[:,:,2] *= 1.1   # Blue
    cool_shading = np.clip(cool_shading, 0, 1)
    cool_lit = albedo * cool_shading
    if residual is not None:
        cool_lit = cool_lit + residual * 0.5
    cool_lit = np.clip(cool_lit, 0, 1)
    np_to_pil(cool_lit).save("intrinsic_relit_cool.png")
    print("   Saved: intrinsic_relit_cool.png")
    
    print("\n" + "="*60)
    print("Done! Output files:")
    print("  - intrinsic_albedo.png (identity/texture)")
    print("  - intrinsic_shading.png (original lighting)")
    print("  - intrinsic_residual.png (specular/reflections)")
    print("  - intrinsic_relit_bright.png")
    print("  - intrinsic_relit_contrast.png")
    print("  - intrinsic_relit_left.png")
    print("  - intrinsic_relit_top.png")
    print("  - intrinsic_relit_warm.png")
    print("  - intrinsic_relit_cool.png")
    print("="*60)


if __name__ == '__main__':
    test_intrinsic()
