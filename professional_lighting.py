#!/usr/bin/env python3
"""
Professional Lighting Correction Module

Provides studio-quality lighting for fitness/training videos:
1. Light Normalization - даже освещение по всему кадру
2. 3D LUT Color Grading - профессиональный цветовой стиль  
3. Skin Tone Enhancement - улучшение тона кожи
4. Virtual Studio Lighting - имитация студийного освещения
5. Adaptive Histogram - автоматическая экспозиция
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class LightingStyle(Enum):
    """Predefined professional lighting styles."""
    NEUTRAL = "neutral"           # Чистый свет без тонирования
    WARM_STUDIO = "warm_studio"   # Тёплый студийный свет
    COOL_STUDIO = "cool_studio"   # Холодный студийный свет
    FITNESS = "fitness"           # Оптимизировано для фитнеса
    BROADCAST = "broadcast"       # ТВ-стиль
    CINEMATIC = "cinematic"       # Кинематографический
    DAYLIGHT = "daylight"         # Естественный дневной свет


def normalize_illumination(image: np.ndarray, 
                          strength: float = 0.7,
                          preserve_color: bool = True) -> np.ndarray:
    """
    Нормализация освещения - убирает неравномерности света.
    Использует локальную адаптацию яркости.
    
    Args:
        image: BGR image
        strength: Сила нормализации (0-1)
        preserve_color: Сохранять цвета при изменении яркости
    """
    if preserve_color:
        # Работаем в LAB для сохранения цвета
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Оценка фонового освещения через сильное размытие
        kernel_size = max(image.shape[0], image.shape[1]) // 4
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_size = max(kernel_size, 31)
        
        background_light = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
        
        # Целевая яркость (средняя по кадру)
        target_brightness = np.mean(l_channel)
        
        # Коррекция локальной яркости
        correction = target_brightness / (background_light + 1e-6)
        correction = np.clip(correction, 0.5, 2.0)  # Ограничиваем коррекцию
        
        # Применяем с заданной силой
        corrected_l = l_channel * (1 - strength + strength * correction)
        corrected_l = np.clip(corrected_l, 0, 255).astype(np.uint8)
        
        lab[:, :, 0] = corrected_l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Простая нормализация по яркости
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel_size = max(image.shape[0], image.shape[1]) // 4
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        target = np.mean(gray)
        correction = target / (background + 1e-6)
        correction = np.clip(correction, 0.5, 2.0)
        
        result = image.astype(np.float32) * correction[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)


def apply_3d_lut_trilinear(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Применяет 3D LUT к изображению с трилинейной интерполяцией.
    
    Args:
        image: BGR image (uint8)
        lut: 3D массив формы (n, n, n, 3), где n - размер LUT (обычно 33 или 64)
    """
    h, w = image.shape[:2]
    lut_size = lut.shape[0]
    
    # Нормализуем координаты в [0, lut_size-1]
    img_f = image.astype(np.float32)
    coords = img_f * (lut_size - 1) / 255.0
    
    # Индексы для интерполяции
    coords_floor = np.floor(coords).astype(np.int32)
    coords_ceil = np.minimum(coords_floor + 1, lut_size - 1)
    
    # Веса для интерполяции
    weights = coords - coords_floor
    
    # Трилинейная интерполяция
    # Получаем 8 угловых значений куба
    c000 = lut[coords_floor[:,:,2], coords_floor[:,:,1], coords_floor[:,:,0]]
    c001 = lut[coords_floor[:,:,2], coords_floor[:,:,1], coords_ceil[:,:,0]]
    c010 = lut[coords_floor[:,:,2], coords_ceil[:,:,1], coords_floor[:,:,0]]
    c011 = lut[coords_floor[:,:,2], coords_ceil[:,:,1], coords_ceil[:,:,0]]
    c100 = lut[coords_ceil[:,:,2], coords_floor[:,:,1], coords_floor[:,:,0]]
    c101 = lut[coords_ceil[:,:,2], coords_floor[:,:,1], coords_ceil[:,:,0]]
    c110 = lut[coords_ceil[:,:,2], coords_ceil[:,:,1], coords_floor[:,:,0]]
    c111 = lut[coords_ceil[:,:,2], coords_ceil[:,:,1], coords_ceil[:,:,0]]
    
    # Интерполяция по каждой оси
    wx = weights[:,:,0:1]
    wy = weights[:,:,1:2]
    wz = weights[:,:,2:3]
    
    c00 = c000 * (1 - wx) + c001 * wx
    c01 = c010 * (1 - wx) + c011 * wx
    c10 = c100 * (1 - wx) + c101 * wx
    c11 = c110 * (1 - wx) + c111 * wx
    
    c0 = c00 * (1 - wy) + c01 * wy
    c1 = c10 * (1 - wy) + c11 * wy
    
    result = c0 * (1 - wz) + c1 * wz
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def create_identity_lut(size: int = 33) -> np.ndarray:
    """Создаёт identity LUT (без изменений)."""
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lut[b, g, r] = [r / (size - 1), g / (size - 1), b / (size - 1)]
    return lut


def create_studio_lut(size: int = 33, style: LightingStyle = LightingStyle.FITNESS) -> np.ndarray:
    """
    Создаёт профессиональный студийный LUT.
    
    Styles:
    - NEUTRAL: Чистый без изменений
    - WARM_STUDIO: Тёплые тона, подходит для комфортного просмотра
    - COOL_STUDIO: Холодные тона, современный вид
    - FITNESS: Оптимизировано для фитнес-видео (контраст + тон кожи)
    - BROADCAST: ТВ-стандарт, хороший баланс
    - CINEMATIC: Кинематографический вид с поднятыми тенями
    - DAYLIGHT: Естественный дневной свет
    """
    lut = create_identity_lut(size)
    
    if style == LightingStyle.NEUTRAL:
        return lut
    
    for b in range(size):
        for g in range(size):
            for r in range(size):
                # Нормализованные значения [0, 1]
                r_val = r / (size - 1)
                g_val = g / (size - 1)
                b_val = b / (size - 1)
                
                # Яркость и насыщенность
                luma = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
                
                if style == LightingStyle.WARM_STUDIO:
                    # Тёплые тона: поднять красный, немного зелёный
                    r_out = np.clip(r_val * 1.08 + 0.02, 0, 1)
                    g_out = np.clip(g_val * 1.03 + 0.01, 0, 1)
                    b_out = np.clip(b_val * 0.95, 0, 1)
                    
                    # Поднять тени
                    shadow_lift = 0.02 * (1 - luma)
                    r_out += shadow_lift
                    g_out += shadow_lift
                    b_out += shadow_lift
                    
                elif style == LightingStyle.COOL_STUDIO:
                    # Холодные тона
                    r_out = np.clip(r_val * 0.95, 0, 1)
                    g_out = np.clip(g_val * 1.02, 0, 1)
                    b_out = np.clip(b_val * 1.08 + 0.02, 0, 1)
                    
                elif style == LightingStyle.FITNESS:
                    # Fitness: контраст + тёплые тона кожи + чёткие тени
                    # S-кривая для контраста
                    contrast_val = lambda x: 1 / (1 + np.exp(-10 * (x - 0.5))) * 0.15 + x * 0.85
                    
                    r_out = contrast_val(r_val) * 1.05
                    g_out = contrast_val(g_val) * 1.02
                    b_out = contrast_val(b_val) * 0.98
                    
                    # Усиление тона кожи (средние красные/жёлтые тона)
                    if 0.3 < r_val < 0.8 and g_val < r_val and b_val < g_val:
                        skin_boost = 0.03 * (1 - abs(luma - 0.5) * 2)
                        r_out += skin_boost
                        g_out += skin_boost * 0.5
                    
                    # Поднять тени
                    shadow_lift = 0.015 * (1 - luma) ** 2
                    r_out += shadow_lift
                    g_out += shadow_lift
                    b_out += shadow_lift
                    
                elif style == LightingStyle.BROADCAST:
                    # ТВ-стандарт: нейтральный с лёгким контрастом
                    mid = 0.5
                    contrast = 1.1
                    
                    r_out = (r_val - mid) * contrast + mid
                    g_out = (g_val - mid) * contrast + mid
                    b_out = (b_val - mid) * contrast + mid
                    
                    # Небольшой тёплый сдвиг
                    r_out = r_out * 1.02
                    b_out = b_out * 0.98
                    
                elif style == LightingStyle.CINEMATIC:
                    # Кинематографический: поднятые тени, teal-orange split
                    # Поднять тени
                    shadow_lift = 0.08 * (1 - luma) ** 1.5
                    
                    # Teal в тенях, orange в светах
                    if luma < 0.5:
                        # Тени - добавить teal (сине-зелёный)
                        teal_amount = (0.5 - luma) * 0.1
                        r_out = r_val - teal_amount * 0.5
                        g_out = g_val + teal_amount * 0.3
                        b_out = b_val + teal_amount * 0.5
                    else:
                        # Света - добавить orange
                        orange_amount = (luma - 0.5) * 0.1
                        r_out = r_val + orange_amount * 0.5
                        g_out = g_val + orange_amount * 0.2
                        b_out = b_val - orange_amount * 0.3
                    
                    r_out += shadow_lift
                    g_out += shadow_lift
                    b_out += shadow_lift
                    
                    # Сжать хайлайты
                    highlight_compress = 0.03 * max(0, luma - 0.7)
                    r_out -= highlight_compress
                    g_out -= highlight_compress
                    b_out -= highlight_compress
                    
                elif style == LightingStyle.DAYLIGHT:
                    # Естественный дневной свет (5500K)
                    r_out = r_val * 1.0
                    g_out = g_val * 1.02
                    b_out = b_val * 1.05
                    
                    # Мягкий контраст
                    mid = 0.5
                    r_out = (r_out - mid) * 1.05 + mid
                    g_out = (g_out - mid) * 1.05 + mid
                    b_out = (b_out - mid) * 1.05 + mid
                else:
                    r_out, g_out, b_out = r_val, g_val, b_val
                
                lut[b, g, r] = [
                    np.clip(r_out, 0, 1),
                    np.clip(g_out, 0, 1),
                    np.clip(b_out, 0, 1)
                ]
    
    return lut


def enhance_skin_tones(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Улучшение тона кожи - делает кожу более здоровой и ровной.
    
    Args:
        image: BGR image
        strength: Сила эффекта (0-1)
    """
    # Конвертируем в HSV для работы с цветом
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Маска для тонов кожи (Hue 0-30 или 330-360 в OpenCV: 0-15 или 165-180)
    skin_mask_low = (h >= 0) & (h <= 25)
    skin_mask_high = (h >= 165) & (h <= 180)
    skin_mask = (skin_mask_low | skin_mask_high) & (s > 20) & (s < 200) & (v > 50)
    
    # Размываем маску для мягкого перехода
    skin_mask_f = skin_mask.astype(np.float32)
    skin_mask_f = cv2.GaussianBlur(skin_mask_f, (21, 21), 0)
    
    # Целевой оттенок кожи (здоровый персиковый)
    target_hue = 15  # Персиковый
    
    # Корректируем оттенок в области кожи
    hue_diff = target_hue - h
    h_corrected = h + hue_diff * skin_mask_f * strength * 0.3
    h_corrected = np.clip(h_corrected, 0, 180)
    
    # Немного снижаем насыщенность для более естественного вида
    s_corrected = s - skin_mask_f * strength * 15
    s_corrected = np.clip(s_corrected, 0, 255)
    
    # Выравниваем яркость кожи
    v_mean_skin = np.mean(v[skin_mask]) if np.any(skin_mask) else 128
    v_diff = v_mean_skin - v
    v_corrected = v + v_diff * skin_mask_f * strength * 0.2
    v_corrected = np.clip(v_corrected, 0, 255)
    
    hsv[:, :, 0] = h_corrected
    hsv[:, :, 1] = s_corrected
    hsv[:, :, 2] = v_corrected
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_virtual_studio_light(image: np.ndarray,
                               key_light_intensity: float = 0.3,
                               fill_light_intensity: float = 0.15,
                               rim_light_intensity: float = 0.1,
                               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Применяет виртуальное студийное освещение.
    
    Имитирует классическую 3-точечную схему освещения:
    - Key light (основной свет) - сверху-слева
    - Fill light (заполняющий) - справа
    - Rim light (контровой) - сзади (по краям)
    
    Args:
        image: BGR image
        key_light_intensity: Интенсивность основного света
        fill_light_intensity: Интенсивность заполняющего света
        rim_light_intensity: Интенсивность контрового света
        mask: Маска объекта (если есть), чтобы свет был только на объекте
    """
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    
    # Key light - сверху слева (тёплый)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    key_light_center = (w * 0.2, h * 0.1)  # Верхний левый угол
    key_dist = np.sqrt((x_coords - key_light_center[0])**2 + 
                       (y_coords - key_light_center[1])**2)
    key_light = 1 - np.clip(key_dist / (max(h, w) * 0.8), 0, 1)
    key_light = key_light ** 0.5  # Мягкий градиент
    
    # Тёплый ключевой свет
    key_color = np.array([0.95, 1.0, 1.05])  # Слегка тёплый
    result[:, :, 0] += key_light[:, :, np.newaxis].squeeze() * key_light_intensity * 255 * key_color[0]
    result[:, :, 1] += key_light[:, :, np.newaxis].squeeze() * key_light_intensity * 255 * key_color[1]
    result[:, :, 2] += key_light[:, :, np.newaxis].squeeze() * key_light_intensity * 255 * key_color[2]
    
    # Fill light - справа (холодный, слабее)
    fill_light_center = (w * 0.9, h * 0.4)
    fill_dist = np.sqrt((x_coords - fill_light_center[0])**2 + 
                        (y_coords - fill_light_center[1])**2)
    fill_light = 1 - np.clip(fill_dist / (max(h, w) * 1.0), 0, 1)
    fill_light = fill_light ** 0.7
    
    # Холодный заполняющий
    fill_color = np.array([1.05, 1.0, 0.95])
    result[:, :, 0] += fill_light * fill_light_intensity * 255 * fill_color[0]
    result[:, :, 1] += fill_light * fill_light_intensity * 255 * fill_color[1]
    result[:, :, 2] += fill_light * fill_light_intensity * 255 * fill_color[2]
    
    # Rim light - по краям (для отделения от фона)
    if mask is not None:
        # Находим края объекта
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0) / 255.0
        
        # Rim light только с одной стороны (сзади-справа)
        rim_gradient = np.linspace(0, 1, w)[np.newaxis, :] * np.ones((h, 1))
        rim_light = edges * rim_gradient * rim_light_intensity
        
        result[:, :, 0] += rim_light * 255 * 1.1
        result[:, :, 1] += rim_light * 255 * 1.05
        result[:, :, 2] += rim_light * 255 * 1.0
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_exposure(image: np.ndarray,
                     target_brightness: float = 128,
                     clip_limit: float = 2.0) -> np.ndarray:
    """
    Адаптивная коррекция экспозиции с CLAHE.
    
    Args:
        image: BGR image
        target_brightness: Целевая средняя яркость (0-255)
        clip_limit: Лимит для CLAHE (выше = больше контраст)
    """
    # Конвертируем в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Текущая яркость
    current_brightness = np.mean(l_channel)
    
    # Сначала глобальная коррекция
    if current_brightness > 0:
        gamma = np.log(target_brightness / 255) / np.log(current_brightness / 255)
        gamma = np.clip(gamma, 0.5, 2.0)  # Ограничиваем коррекцию
        l_corrected = np.power(l_channel / 255.0, gamma) * 255
    else:
        l_corrected = l_channel.astype(np.float32)
    
    lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
    
    # CLAHE для локального контраста
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def professional_lighting_pipeline(image: np.ndarray,
                                   mask: Optional[np.ndarray] = None,
                                   style: LightingStyle = LightingStyle.FITNESS,
                                   normalize_light: bool = True,
                                   enhance_skin: bool = True,
                                   virtual_studio: bool = False,
                                   adaptive_exposure_enabled: bool = True) -> np.ndarray:
    """
    Полный пайплайн профессионального освещения.
    
    Args:
        image: BGR image
        mask: Маска человека (опционально)
        style: Стиль LUT
        normalize_light: Нормализовать неравномерное освещение
        enhance_skin: Улучшить тон кожи
        virtual_studio: Добавить виртуальный студийный свет
        adaptive_exposure_enabled: Автоматическая экспозиция
    """
    result = image.copy()
    
    # 1. Адаптивная экспозиция
    if adaptive_exposure_enabled:
        result = adaptive_exposure(result, target_brightness=120, clip_limit=1.5)
    
    # 2. Нормализация освещения
    if normalize_light:
        result = normalize_illumination(result, strength=0.5)
    
    # 3. Виртуальный студийный свет
    if virtual_studio and mask is not None:
        result = apply_virtual_studio_light(
            result, 
            key_light_intensity=0.2,
            fill_light_intensity=0.1,
            rim_light_intensity=0.08,
            mask=mask
        )
    
    # 4. 3D LUT цветокоррекция
    if style != LightingStyle.NEUTRAL:
        lut = create_studio_lut(33, style)
        result = apply_3d_lut_trilinear(result, lut)
    
    # 5. Улучшение тона кожи
    if enhance_skin:
        result = enhance_skin_tones(result, strength=0.3)
    
    return result


# Quick presets for common use cases
def apply_fitness_lighting(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Быстрый пресет для фитнес-видео."""
    return professional_lighting_pipeline(
        image, mask,
        style=LightingStyle.FITNESS,
        normalize_light=True,
        enhance_skin=True,
        virtual_studio=mask is not None,
        adaptive_exposure_enabled=True
    )


def apply_broadcast_lighting(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Быстрый пресет для ТВ-стиля."""
    return professional_lighting_pipeline(
        image, mask,
        style=LightingStyle.BROADCAST,
        normalize_light=True,
        enhance_skin=True,
        virtual_studio=False,
        adaptive_exposure_enabled=True
    )


def apply_cinematic_lighting(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Быстрый пресет для кинематографического стиля."""
    return professional_lighting_pipeline(
        image, mask,
        style=LightingStyle.CINEMATIC,
        normalize_light=False,  # Кино часто имеет неравномерный свет
        enhance_skin=False,
        virtual_studio=False,
        adaptive_exposure_enabled=False
    )


if __name__ == "__main__":
    # Тест
    import argparse
    
    parser = argparse.ArgumentParser(description='Test professional lighting')
    parser.add_argument('--input', '-i', required=True, help='Input image')
    parser.add_argument('--output', '-o', default='output_lighting', help='Output prefix')
    parser.add_argument('--style', '-s', default='fitness',
                       choices=['neutral', 'warm_studio', 'cool_studio', 'fitness', 
                               'broadcast', 'cinematic', 'daylight'])
    args = parser.parse_args()
    
    # Загружаем изображение
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Cannot load {args.input}")
        exit(1)
    
    # Сохраняем оригинал
    cv2.imwrite(f'{args.output}_original.jpg', img)
    print(f"Saved: {args.output}_original.jpg")
    
    # Тестируем каждый стиль
    styles = [
        LightingStyle.NEUTRAL,
        LightingStyle.WARM_STUDIO,
        LightingStyle.COOL_STUDIO,
        LightingStyle.FITNESS,
        LightingStyle.BROADCAST,
        LightingStyle.CINEMATIC,
        LightingStyle.DAYLIGHT,
    ]
    
    for style in styles:
        result = professional_lighting_pipeline(
            img, 
            style=style,
            normalize_light=True,
            enhance_skin=True
        )
        output_name = f'{args.output}_{style.value}.jpg'
        cv2.imwrite(output_name, result)
        print(f"Saved: {output_name}")
    
    print("\nDone! Compare the results.")
