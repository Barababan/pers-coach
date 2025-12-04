#!/usr/bin/env python3
"""
Smart Mirror Quality Processing

Оптимизировано для интерактивного зеркала тренировок:
- Чистый равномерный фон без градиентов
- Хорошо освещённый человек с видимыми деталями
- Естественные цвета кожи
- Чёткие контуры для сравнения позы
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def fix_glow_artifact(image: np.ndarray, mask: np.ndarray, 
                      bg_color: Tuple[int, int, int] = (128, 128, 128),
                      glow_threshold: int = 20) -> np.ndarray:
    """
    Убирает артефакт свечения (glow) вокруг человека.
    Свечение появляется когда полупрозрачные пиксели маски 
    смешиваются с фоном неправильно.
    
    Args:
        image: BGR изображение с уже примененной маской
        mask: Маска человека (0-1 float или 0-255 uint8)
        bg_color: Целевой цвет фона
        glow_threshold: Порог детекции свечения
    """
    h, w = image.shape[:2]
    
    # Нормализуем маску
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Находим область вокруг человека (потенциальное свечение)
    # Это пиксели где маска не 0 и не 1
    semi_transparent = (mask_f > 0.01) & (mask_f < 0.99)
    
    # Расширяем на область вокруг маски
    mask_uint8 = (mask_f * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
    outer_region = (dilated > 128) & (mask_uint8 < 200)
    
    # Создаём чистый фон
    result = image.copy()
    bg_array = np.array(bg_color, dtype=np.uint8)
    
    # В области свечения заменяем на чистый фон
    # Но сохраняем пиксели которые сильно отличаются от фона (часть человека)
    image_f = image.astype(np.float32)
    bg_f = np.array(bg_color, dtype=np.float32)
    
    # Разница от фона
    diff_from_bg = np.sqrt(np.sum((image_f - bg_f) ** 2, axis=-1))
    
    # Свечение = в outer_region, но близко к фону по цвету
    glow_mask = outer_region & (diff_from_bg < glow_threshold * 3)
    
    # Заменяем свечение на чистый фон
    result[glow_mask] = bg_array
    
    return result


def ensure_clean_background(image: np.ndarray, mask: np.ndarray,
                           bg_color: Tuple[int, int, int] = (128, 128, 128),
                           edge_blend: int = 2) -> np.ndarray:
    """
    Гарантирует чистый равномерный фон без градиентов.
    
    Args:
        image: Исходное BGR изображение (до применения маски)
        mask: Маска человека (float 0-1)
        bg_color: Цвет фона
        edge_blend: Размер зоны сглаживания на краях (пиксели)
    """
    h, w = image.shape[:2]
    
    # Нормализуем маску
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Делаем маску более бинарной (убираем полупрозрачность)
    # но сохраняем мягкие края
    mask_sharp = np.where(mask_f > 0.5, 1.0, 0.0).astype(np.float32)
    
    # Сглаживаем края
    if edge_blend > 0:
        kernel_size = edge_blend * 2 + 1
        mask_sharp = cv2.GaussianBlur(mask_sharp, (kernel_size, kernel_size), 0)
    
    # Применяем маску
    mask_3d = np.stack([mask_sharp] * 3, axis=-1)
    bg = np.full((h, w, 3), bg_color, dtype=np.float32)
    
    result = mask_3d * image.astype(np.float32) + (1 - mask_3d) * bg
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_person_visibility(image: np.ndarray, mask: np.ndarray,
                             target_brightness: float = 140,
                             contrast_boost: float = 1.15,
                             saturation_boost: float = 1.1) -> np.ndarray:
    """
    Улучшает видимость человека для тренировок.
    Поднимает яркость и контраст только на человеке.
    
    Args:
        image: BGR изображение
        mask: Маска человека (float 0-1)
        target_brightness: Целевая яркость (0-255)
        contrast_boost: Усиление контраста
        saturation_boost: Усиление насыщенности
    """
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Работаем только с областью человека
    person_mask = mask_f > 0.5
    
    if not np.any(person_mask):
        return image
    
    result = image.copy()
    
    # 1. Коррекция яркости в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    # Текущая средняя яркость человека
    current_brightness = np.mean(l_channel[person_mask])
    
    if current_brightness > 0:
        # Гамма-коррекция для достижения целевой яркости
        gamma = np.log(target_brightness / 255) / np.log(current_brightness / 255 + 1e-6)
        gamma = np.clip(gamma, 0.5, 2.0)
        
        # Применяем только к человеку
        l_corrected = np.power(l_channel / 255.0, gamma) * 255
        l_channel_new = np.where(person_mask, l_corrected, l_channel)
        lab[:, :, 0] = np.clip(l_channel_new, 0, 255)
    
    # 2. Усиление контраста (S-кривая)
    l_channel = lab[:, :, 0]
    l_mean = np.mean(l_channel[person_mask])
    l_contrasted = (l_channel - l_mean) * contrast_boost + l_mean
    lab[:, :, 0] = np.where(person_mask, np.clip(l_contrasted, 0, 255), lab[:, :, 0])
    
    # 3. Усиление насыщенности (a и b каналы)
    lab[:, :, 1] = np.where(person_mask[:, :, np.newaxis].squeeze() if person_mask.ndim == 2 else person_mask,
                           np.clip((lab[:, :, 1] - 128) * saturation_boost + 128, 0, 255),
                           lab[:, :, 1])
    lab[:, :, 2] = np.where(person_mask[:, :, np.newaxis].squeeze() if person_mask.ndim == 2 else person_mask,
                           np.clip((lab[:, :, 2] - 128) * saturation_boost + 128, 0, 255),
                           lab[:, :, 2])
    
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return result


def enhance_skin_tones_smart_mirror(image: np.ndarray, mask: np.ndarray,
                                    warmth: float = 0.03) -> np.ndarray:
    """
    Улучшение тона кожи для smart mirror.
    Делает кожу более естественной и здоровой.
    
    Args:
        image: BGR изображение
        mask: Маска человека
        warmth: Степень "тепла" (0-0.1)
    """
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    person_mask = mask_f > 0.5
    
    result = image.astype(np.float32)
    
    # Добавляем немного тепла (увеличиваем R, уменьшаем B)
    result[:, :, 2] = np.where(person_mask, 
                               np.clip(result[:, :, 2] * (1 + warmth), 0, 255),
                               result[:, :, 2])
    result[:, :, 0] = np.where(person_mask,
                               np.clip(result[:, :, 0] * (1 - warmth * 0.5), 0, 255),
                               result[:, :, 0])
    
    return result.astype(np.uint8)


def sharpen_edges(image: np.ndarray, mask: np.ndarray, 
                  amount: float = 0.5) -> np.ndarray:
    """
    Усиливает чёткость краёв человека для лучшего восприятия позы.
    
    Args:
        image: BGR изображение
        mask: Маска человека
        amount: Сила эффекта (0-1)
    """
    # Unsharp mask
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    
    # Применяем только к краям человека
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Находим края маски
    edges = cv2.Canny((mask_f * 255).astype(np.uint8), 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges_f = cv2.GaussianBlur(edges.astype(np.float32), (11, 11), 0) / 255.0
    
    # Смешиваем
    edges_3d = np.stack([edges_f] * 3, axis=-1)
    result = edges_3d * sharpened.astype(np.float32) + (1 - edges_3d) * image.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_before_cutout(image_bgr: np.ndarray,
                          mask: np.ndarray,
                          target_brightness: float = 145,
                          contrast: float = 1.12,
                          saturation: float = 1.08,
                          warmth: float = 0.02) -> np.ndarray:
    """
    Улучшает изображение ДО вырезания фона.
    ВАЖНО: Применяется ТОЛЬКО к области человека (по маске),
    чтобы яркий фон не влиял на коррекцию.
    
    Args:
        image_bgr: BGR изображение (полное, с оригинальным фоном)
        mask: Маска человека (float 0-1)
        target_brightness: Целевая яркость
        contrast: Контраст
        saturation: Насыщенность
        warmth: Теплота
    """
    # Нормализуем маску
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Находим область человека для расчёта статистик
    person_mask = mask_f > 0.5
    
    if not np.any(person_mask):
        return image_bgr
    
    result = image_bgr.copy()
    
    # 1. Яркость и контраст в LAB пространстве
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    # Текущая средняя яркость ТОЛЬКО ЧЕЛОВЕКА (не всего изображения!)
    current_brightness = np.mean(l_channel[person_mask])
    
    if current_brightness > 0 and abs(current_brightness - target_brightness) > 5:
        # Гамма-коррекция только для человека
        gamma = np.log(target_brightness / 255) / np.log(current_brightness / 255 + 1e-6)
        gamma = np.clip(gamma, 0.7, 1.5)  # Более консервативные границы
        
        # Применяем ТОЛЬКО к области человека
        l_corrected = np.power(l_channel / 255.0, gamma) * 255
        l_channel = np.where(person_mask, l_corrected, l_channel)
    
    # Контраст - тоже только для человека
    l_mean = np.mean(l_channel[person_mask])
    l_contrasted = (l_channel - l_mean) * contrast + l_mean
    lab[:, :, 0] = np.where(person_mask, np.clip(l_contrasted, 0, 255), l_channel)
    
    # 2. Насыщенность - только для человека
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    lab[:, :, 1] = np.where(person_mask, np.clip((a_channel - 128) * saturation + 128, 0, 255), a_channel)
    lab[:, :, 2] = np.where(person_mask, np.clip((b_channel - 128) * saturation + 128, 0, 255), b_channel)
    
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    
    # 3. Теплота - только для человека
    if warmth > 0:
        r_channel = result[:, :, 2]
        b_channel_bgr = result[:, :, 0]
        result[:, :, 2] = np.where(person_mask, np.clip(r_channel * (1 + warmth), 0, 255), r_channel)
        result[:, :, 0] = np.where(person_mask, np.clip(b_channel_bgr * (1 - warmth * 0.5), 0, 255), b_channel_bgr)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def smart_mirror_pipeline(image_rgb: np.ndarray, 
                          mask: np.ndarray,
                          bg_color: Tuple[int, int, int] = (128, 128, 128),
                          target_brightness: float = 145,
                          contrast: float = 1.12,
                          saturation: float = 1.08,
                          warmth: float = 0.02,
                          sharpen: float = 0.3,
                          edge_blend: int = 2) -> np.ndarray:
    """
    Полный пайплайн обработки для Smart Mirror.
    
    ВАЖНО: Порядок операций критичен для качества!
    1. Сначала улучшаем освещение ВСЕГО изображения (до вырезания)
    2. Потом вырезаем с чистой маской
    Это предотвращает артефакты свечения по краям.
    
    Args:
        image_rgb: RGB изображение человека (с оригинальным фоном!)
        mask: Маска человека (float 0-1)
        bg_color: Цвет фона RGB
        target_brightness: Целевая яркость человека
        contrast: Усиление контраста
        saturation: Усиление насыщенности
        warmth: Теплота цветов
        sharpen: Резкость краёв
        edge_blend: Сглаживание краёв маски
    
    Returns:
        BGR изображение готовое для отображения
    """
    # Конвертируем в BGR для OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # ========================================
    # ШАГ 1: УЛУЧШАЕМ ОСВЕЩЕНИЕ ЧЕЛОВЕКА
    # ========================================
    # Применяем коррекцию ТОЛЬКО к области человека по маске,
    # чтобы яркий фон (окна и т.п.) не влиял на расчёт яркости
    result = enhance_before_cutout(
        image_bgr,
        mask,  # Передаём маску!
        target_brightness=target_brightness,
        contrast=contrast,
        saturation=saturation,
        warmth=warmth
    )
    
    # ========================================
    # ШАГ 2: ВЫРЕЗАЕМ С ЧИСТОЙ МАСКОЙ
    # ========================================
    # bg_color в RGB, конвертируем в BGR
    bg_bgr = (bg_color[2], bg_color[1], bg_color[0])
    result = ensure_clean_background(result, mask, bg_color=bg_bgr, edge_blend=edge_blend)
    
    # ========================================  
    # ШАГ 3: ФИНАЛЬНЫЕ ШТРИХИ (только если нужно)
    # ========================================
    # Усиливаем чёткость краёв
    if sharpen > 0:
        result = sharpen_edges(result, mask, amount=sharpen)
    
    return result


def process_for_smart_mirror(image_rgb: np.ndarray,
                             mask: np.ndarray,
                             preset: str = 'balanced') -> np.ndarray:
    """
    Простой интерфейс для smart mirror с пресетами.
    
    Presets:
    - 'bright': Яркое изображение, хорошо видны детали
    - 'balanced': Сбалансированное (рекомендуется)
    - 'natural': Максимально естественное
    - 'high_contrast': Высокий контраст для чётких силуэтов
    """
    presets = {
        'bright': {
            'target_brightness': 160,
            'contrast': 1.15,
            'saturation': 1.1,
            'warmth': 0.03,
            'sharpen': 0.4,
            'edge_blend': 2
        },
        'balanced': {
            'target_brightness': 145,
            'contrast': 1.12,
            'saturation': 1.08,
            'warmth': 0.02,
            'sharpen': 0.3,
            'edge_blend': 2
        },
        'natural': {
            'target_brightness': 130,
            'contrast': 1.05,
            'saturation': 1.02,
            'warmth': 0.01,
            'sharpen': 0.2,
            'edge_blend': 3
        },
        'high_contrast': {
            'target_brightness': 140,
            'contrast': 1.25,
            'saturation': 1.15,
            'warmth': 0.02,
            'sharpen': 0.5,
            'edge_blend': 1
        }
    }
    
    params = presets.get(preset, presets['balanced'])
    
    return smart_mirror_pipeline(
        image_rgb, mask,
        bg_color=(128, 128, 128),
        **params
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Smart Mirror processing')
    parser.add_argument('--input', '-i', required=True, help='Input image with person on gray background')
    parser.add_argument('--output', '-o', default='smart_mirror_output', help='Output prefix')
    parser.add_argument('--preset', '-p', default='balanced', 
                       choices=['bright', 'balanced', 'natural', 'high_contrast'])
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Cannot load {args.input}")
        exit(1)
    
    # For testing, create a simple mask from the gray background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect gray background (around 128)
    bg_mask = (gray > 100) & (gray < 156)
    person_mask = ~bg_mask
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    person_mask = cv2.morphologyEx(person_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
    person_mask = person_mask.astype(np.float32) / 255.0
    
    # Convert to RGB for pipeline
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Test all presets
    presets = ['bright', 'balanced', 'natural', 'high_contrast']
    
    for preset in presets:
        result = process_for_smart_mirror(img_rgb, person_mask, preset=preset)
        output_path = f'{args.output}_{preset}.jpg'
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")
    
    print("\nDone!")
