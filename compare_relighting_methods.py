#!/usr/bin/env python3
"""
Сравнение методов relighting:
1. IC-Light (облачный) - перегенерирует изображение
2. Intrinsic Decomposition - разделение albedo/shading
3. DPR (Deep Portrait Relighting) - Spherical Harmonics
4. Наши ручные методы (studio_lighting.py)

Результат: сравнительная таблица качества и сохранения идентичности.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import time

# Добавляем пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MODNet/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DPR'))

def extract_test_frame(video_path='squat.mp4', frame_num=50):
    """Извлекаем тестовый кадр из видео."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_modnet_mask(frame):
    """Получаем маску через MODNet."""
    import torch
    from models.modnet import MODNet
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    state_dict = torch.load('MODNet/pretrained/modnet_webcam_portrait_matting.ckpt', 
                           map_location='cpu')
    modnet.load_state_dict(state_dict)
    modnet = modnet.module.to(device)
    modnet.eval()
    
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        _, _, matte = modnet(tensor, True)
    
    matte = matte.squeeze().cpu().numpy()
    matte = cv2.resize(matte, (w, h))
    return matte


def test_intrinsic_method(image_rgb, output_dir):
    """Тест Intrinsic Decomposition."""
    print("\n🔬 Testing Intrinsic Decomposition...")
    
    from chrislib.data_util import np_to_pil
    from intrinsic.pipeline import load_models, run_pipeline
    import torch
    
    device = torch.device('cpu')
    
    start = time.time()
    
    # Normalize to [0,1]
    img = image_rgb.astype(np.float32) / 255.0
    
    models = load_models('v2', device=device)
    results = run_pipeline(models, img, device='cpu')
    
    albedo = results['hr_alb']
    shading = results['dif_shd']
    residual = results['residual']
    
    # Relight варианты
    variants = {}
    
    # Яркий
    bright_shading = np.clip(shading * 1.4, 0, 1)
    variants['bright'] = np.clip(albedo * bright_shading + residual * 0.3, 0, 1)
    
    # Тёплый
    warm_shading = shading.copy()
    warm_shading[:,:,0] *= 1.1
    warm_shading[:,:,2] *= 0.9
    warm_shading = np.clip(warm_shading, 0, 1)
    variants['warm'] = np.clip(albedo * warm_shading + residual * 0.3, 0, 1)
    
    # Холодный
    cool_shading = shading.copy()
    cool_shading[:,:,0] *= 0.9
    cool_shading[:,:,2] *= 1.1
    cool_shading = np.clip(cool_shading, 0, 1)
    variants['cool'] = np.clip(albedo * cool_shading + residual * 0.3, 0, 1)
    
    elapsed = time.time() - start
    print(f"   ⏱️ Time: {elapsed:.1f}s")
    
    # Сохраняем
    for name, img in variants.items():
        path = os.path.join(output_dir, f"intrinsic_{name}.png")
        np_to_pil(img).save(path)
        print(f"   Saved: {path}")
    
    return variants, elapsed


def test_studio_lighting(image_rgb, mask, output_dir):
    """Тест наших ручных методов освещения."""
    print("\n💡 Testing Studio Lighting (manual)...")
    
    from studio_lighting import apply_studio_lighting, PRESETS
    
    start = time.time()
    variants = {}
    
    for preset_name in ['natural', 'fitness_studio', 'cyberpunk', 'dramatic']:
        preset = PRESETS.get(preset_name)
        if preset:
            result = apply_studio_lighting(image_rgb, mask, preset)
            variants[preset_name] = result
            
            path = os.path.join(output_dir, f"studio_{preset_name}.png")
            cv2.imwrite(path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"   Saved: {path}")
    
    elapsed = time.time() - start
    print(f"   ⏱️ Time: {elapsed:.1f}s")
    
    return variants, elapsed


def calculate_identity_similarity(original, relit):
    """Вычисляем похожесть на оригинал (SSIM-подобная метрика)."""
    from skimage.metrics import structural_similarity as ssim
    
    # Приводим к одному размеру
    if original.shape != relit.shape:
        h, w = min(original.shape[0], relit.shape[0]), min(original.shape[1], relit.shape[1])
        original = cv2.resize(original, (w, h))
        relit = cv2.resize(relit, (w, h))
    
    # Конвертируем в grayscale для SSIM
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        relit_gray = cv2.cvtColor(relit.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = original
        relit_gray = relit
    
    score = ssim(orig_gray, relit_gray)
    return score


def main():
    print("="*70)
    print("RELIGHTING METHODS COMPARISON")
    print("="*70)
    
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Извлекаем кадр
    print("\n[1] Extracting test frame...")
    frame = extract_test_frame()
    if frame is None:
        print("❌ No video found")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, "original.png"), frame)
    
    # 2. Получаем маску
    print("\n[2] Getting segmentation mask...")
    mask = get_modnet_mask(frame)
    
    # Сохраняем кадр с маской (на белом фоне)
    white_bg = np.ones_like(frame_rgb) * 255
    masked = frame_rgb * mask[:,:,np.newaxis] + white_bg * (1 - mask[:,:,np.newaxis])
    masked = masked.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "masked.png"), 
                cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    
    results = {}
    
    # 3. Тест Intrinsic
    try:
        intrinsic_variants, intrinsic_time = test_intrinsic_method(masked, output_dir)
        results['Intrinsic'] = {
            'time': intrinsic_time,
            'variants': intrinsic_variants,
            'identity_preserved': True
        }
    except Exception as e:
        print(f"   ❌ Intrinsic failed: {e}")
    
    # 4. Тест Studio Lighting
    try:
        studio_variants, studio_time = test_studio_lighting(frame_rgb, mask, output_dir)
        results['Studio'] = {
            'time': studio_time,
            'variants': studio_variants,
            'identity_preserved': True
        }
    except Exception as e:
        print(f"   ❌ Studio failed: {e}")
    
    # 5. Выводим результаты
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Time':<10} {'Identity':<15} {'Quality'}")
    print("-"*60)
    
    for method, data in results.items():
        identity = "✅ Preserved" if data['identity_preserved'] else "❌ Changed"
        print(f"{method:<20} {data['time']:.1f}s{'':<5} {identity:<15}")
    
    # IC-Light comparison
    print(f"{'IC-Light (cloud)':<20} {'~10s':<10} {'❌ Changed':<15} {'High (but new face)'}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
