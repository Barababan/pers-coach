#!/usr/bin/env python3
"""
Studio Lighting - Профессиональное студийное освещение для видео фитнеса.

Симулирует реальные источники света с направлением, цветом и интенсивностью.
Включает модные RGB-схемы освещения (синий/фиолетовый, неон и т.д.)

Основано на принципах 3D relighting без тяжёлых нейросетей.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Light:
    """Описание источника света."""
    direction: Tuple[float, float, float]  # Нормализованный вектор направления (x, y, z)
    color: Tuple[int, int, int]  # RGB цвет
    intensity: float  # Интенсивность 0-1
    softness: float = 0.5  # Мягкость света 0-1 (0=жёсткий, 1=очень мягкий)


class LightingPreset(Enum):
    """Пресеты студийного освещения."""
    NATURAL = "natural"                  # Естественный дневной свет
    FITNESS_STUDIO = "fitness_studio"    # Классическая фитнес-студия
    CYBERPUNK = "cyberpunk"              # Синий + фиолетовый (модный)
    NEON_GYM = "neon_gym"                # Неоновый спортзал
    SUNSET_WARM = "sunset_warm"          # Тёплый закатный
    COOL_PROFESSIONAL = "cool_pro"       # Холодный профессиональный
    DRAMATIC = "dramatic"                # Драматичный контрастный
    RGB_TRICOLOR = "rgb_tricolor"        # Трёхцветный RGB


# Пресеты освещения
LIGHTING_PRESETS = {
    LightingPreset.NATURAL: [
        Light(direction=(0, 0.3, 1), color=(255, 250, 240), intensity=0.7, softness=0.8),
        Light(direction=(0, 1, 0.2), color=(200, 220, 255), intensity=0.3, softness=0.9),  # Небо
    ],
    
    LightingPreset.FITNESS_STUDIO: [
        Light(direction=(0, 0.5, 1), color=(255, 255, 255), intensity=0.6, softness=0.6),  # Фронт
        Light(direction=(-0.7, 0.3, 0.5), color=(255, 245, 235), intensity=0.3, softness=0.7),  # Лево
        Light(direction=(0.7, 0.3, 0.5), color=(255, 245, 235), intensity=0.3, softness=0.7),  # Право
    ],
    
    LightingPreset.CYBERPUNK: [
        Light(direction=(-0.8, 0.2, 0.5), color=(0, 150, 255), intensity=0.5, softness=0.4),  # Синий слева
        Light(direction=(0.8, 0.2, 0.5), color=(200, 0, 255), intensity=0.5, softness=0.4),  # Фиолетовый справа
        Light(direction=(0, 0.5, 1), color=(255, 255, 255), intensity=0.3, softness=0.6),  # Слабый фронт
    ],
    
    LightingPreset.NEON_GYM: [
        Light(direction=(-0.7, 0, 0.5), color=(255, 0, 100), intensity=0.4, softness=0.3),  # Розовый слева
        Light(direction=(0.7, 0, 0.5), color=(0, 255, 200), intensity=0.4, softness=0.3),  # Бирюзовый справа
        Light(direction=(0, 0.8, 0.3), color=(255, 255, 255), intensity=0.4, softness=0.5),  # Белый сверху
    ],
    
    LightingPreset.SUNSET_WARM: [
        Light(direction=(-0.5, 0.3, 0.8), color=(255, 180, 100), intensity=0.6, softness=0.5),  # Тёплый ключевой
        Light(direction=(0.5, 0.5, 0.3), color=(255, 200, 150), intensity=0.3, softness=0.7),  # Заполняющий
        Light(direction=(0, 0.2, -0.8), color=(255, 150, 80), intensity=0.2, softness=0.4),  # Контровой
    ],
    
    LightingPreset.COOL_PROFESSIONAL: [
        Light(direction=(0, 0.4, 1), color=(240, 248, 255), intensity=0.6, softness=0.6),  # Холодный фронт
        Light(direction=(-0.6, 0.3, 0.5), color=(220, 235, 255), intensity=0.35, softness=0.7),
        Light(direction=(0.6, 0.3, 0.5), color=(220, 235, 255), intensity=0.35, softness=0.7),
    ],
    
    LightingPreset.DRAMATIC: [
        Light(direction=(-0.8, 0.5, 0.3), color=(255, 240, 220), intensity=0.7, softness=0.3),  # Жёсткий ключевой
        Light(direction=(0.3, 0.2, 0.5), color=(100, 120, 150), intensity=0.2, softness=0.8),  # Слабое заполнение
    ],
    
    LightingPreset.RGB_TRICOLOR: [
        Light(direction=(-0.8, 0.3, 0.5), color=(255, 50, 50), intensity=0.35, softness=0.4),  # Красный
        Light(direction=(0, 0.8, 0.3), color=(50, 255, 50), intensity=0.35, softness=0.4),  # Зелёный сверху
        Light(direction=(0.8, 0.3, 0.5), color=(50, 50, 255), intensity=0.35, softness=0.4),  # Синий
        Light(direction=(0, 0.3, 1), color=(200, 200, 200), intensity=0.25, softness=0.6),  # Нейтральный фронт
    ],
}


def estimate_surface_normals(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Оценивает нормали поверхности из изображения.
    Использует градиенты яркости как приближение к нормалям.
    
    Для полноценного relighting нужна depth map или 3D модель,
    но этот метод даёт неплохой результат для 2D изображений.
    """
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Вычисляем градиенты (приближение к dx, dy нормали)
    # Используем Sobel для более гладких градиентов
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5) * 2
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5) * 2
    
    # Нормаль направлена "наружу" из поверхности
    # Для 2D приближения: normal = (-dx, -dy, 1) normalized
    h, w = gray.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :, 0] = -grad_x  # X компонент
    normals[:, :, 1] = -grad_y  # Y компонент  
    normals[:, :, 2] = 1.0      # Z компонент (направлен к камере)
    
    # Нормализуем
    norm = np.sqrt(np.sum(normals ** 2, axis=-1, keepdims=True)) + 1e-8
    normals = normals / norm
    
    return normals


def estimate_depth_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Простая оценка глубины из маски.
    Центр маски = ближе к камере, края = дальше.
    """
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Distance transform - расстояние от краёв
    mask_uint8 = (mask_f * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    
    # Нормализуем
    if dist.max() > 0:
        dist = dist / dist.max()
    
    # Размываем для гладкости
    depth = cv2.GaussianBlur(dist, (21, 21), 0)
    
    return depth


def apply_directional_light(image_bgr: np.ndarray, 
                           normals: np.ndarray,
                           depth: np.ndarray,
                           mask: np.ndarray,
                           light: Light) -> np.ndarray:
    """
    Применяет направленный источник света к изображению.
    
    Использует упрощённую модель освещения:
    - Lambertian diffuse (dot product нормали и направления света)
    - Учёт глубины для более реалистичного результата
    """
    h, w = image_bgr.shape[:2]
    
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Нормализуем направление света
    light_dir = np.array(light.direction, dtype=np.float32)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    
    # Вычисляем dot product (N · L) - насколько поверхность обращена к свету
    # normals shape: (h, w, 3), light_dir shape: (3,)
    n_dot_l = np.sum(normals * light_dir, axis=-1)
    
    # Clamp к [0, 1] - отрицательные значения = тень
    n_dot_l = np.clip(n_dot_l, 0, 1)
    
    # Мягкость света - размываем n_dot_l
    if light.softness > 0:
        blur_size = int(light.softness * 50) * 2 + 1
        n_dot_l = cv2.GaussianBlur(n_dot_l, (blur_size, blur_size), 0)
    
    # Модулируем глубиной (опционально - даёт объём)
    # Ближние части получают чуть больше света
    depth_factor = 0.7 + 0.3 * depth
    n_dot_l = n_dot_l * depth_factor
    
    # Применяем цвет и интенсивность света
    light_color = np.array(light.color, dtype=np.float32) / 255.0
    light_contribution = n_dot_l[:, :, np.newaxis] * light_color * light.intensity
    
    # Применяем только к области маски
    mask_3d = mask_f[:, :, np.newaxis]
    
    return light_contribution * mask_3d


def apply_studio_lighting(image_bgr: np.ndarray,
                         mask: np.ndarray,
                         preset: LightingPreset = LightingPreset.CYBERPUNK,
                         ambient: float = 0.15,
                         strength: float = 1.0) -> np.ndarray:
    """
    Применяет студийное освещение к изображению.
    
    Args:
        image_bgr: BGR изображение
        mask: Маска человека (0-1 float или 0-255 uint8)
        preset: Пресет освещения
        ambient: Уровень окружающего света (0-1)
        strength: Сила эффекта (0-1, где 0=без эффекта, 1=полный эффект)
    
    Returns:
        BGR изображение с применённым освещением
    """
    h, w = image_bgr.shape[:2]
    
    # Нормализуем маску
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    
    # Оцениваем нормали и глубину
    normals = estimate_surface_normals(image_bgr, mask_f)
    depth = estimate_depth_from_mask(mask_f)
    
    # Получаем источники света для пресета
    lights = LIGHTING_PRESETS.get(preset, LIGHTING_PRESETS[LightingPreset.FITNESS_STUDIO])
    
    # Вычисляем вклад каждого источника света
    total_light = np.zeros((h, w, 3), dtype=np.float32)
    
    for light in lights:
        light_contribution = apply_directional_light(
            image_bgr, normals, depth, mask_f, light
        )
        total_light += light_contribution
    
    # Добавляем ambient (окружающий свет)
    person_mask = mask_f > 0.3
    ambient_light = np.zeros((h, w, 3), dtype=np.float32)
    ambient_light[person_mask] = ambient
    total_light += ambient_light[:, :, np.newaxis].squeeze() if ambient_light.ndim == 4 else ambient_light
    
    # Нормализуем изображение
    image_f = image_bgr.astype(np.float32) / 255.0
    
    # Применяем освещение
    # Используем multiplicative blending с усилением
    lit_image = image_f * (0.3 + total_light * 1.5)  # 0.3 = базовый уровень
    
    # Смешиваем с оригиналом по strength
    mask_3d = mask_f[:, :, np.newaxis]
    result = image_f * (1 - mask_3d * strength) + lit_image * mask_3d * strength
    
    # Clamp и конвертируем обратно
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result


def create_custom_lighting(lights: List[Light]) -> List[Light]:
    """
    Создаёт кастомную схему освещения.
    
    Пример использования:
    lights = create_custom_lighting([
        Light((-0.8, 0.2, 0.5), (0, 100, 255), 0.5),  # Синий слева
        Light((0.8, 0.2, 0.5), (255, 0, 100), 0.5),   # Розовый справа
    ])
    """
    return lights


def visualize_lighting_setup(preset: LightingPreset, size: int = 400) -> np.ndarray:
    """
    Визуализирует схему освещения (вид сверху).
    Полезно для понимания расположения источников.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    
    # Рисуем "человека" в центре
    cv2.circle(img, (center, center), 30, (100, 100, 100), -1)
    cv2.putText(img, "Person", (center-25, center+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    lights = LIGHTING_PRESETS.get(preset, [])
    
    for i, light in enumerate(lights):
        # Конвертируем направление в позицию на схеме
        # direction указывает ОТ источника К объекту, инвертируем
        dx, dy, dz = light.direction
        
        # Позиция источника (вид сверху, z игнорируем для визуализации)
        lx = int(center - dx * 150)  # Инвертируем x
        ly = int(center - dy * 150)  # Инвертируем y
        
        # Цвет источника (BGR)
        color = (light.color[2], light.color[1], light.color[0])
        
        # Размер зависит от интенсивности
        radius = int(10 + light.intensity * 20)
        
        # Рисуем источник
        cv2.circle(img, (lx, ly), radius, color, -1)
        
        # Линия к объекту
        cv2.line(img, (lx, ly), (center, center), color, 1)
        
        # Подпись
        cv2.putText(img, f"L{i+1}", (lx-10, ly-radius-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Название пресета
    cv2.putText(img, preset.value, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img


def test_all_presets(image_path: str, mask_path: str = None):
    """Тестирует все пресеты освещения на изображении."""
    import os
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить {image_path}")
        return
    
    # Загружаем или создаём маску
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
    else:
        # Простая маска - центральный прямоугольник
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h//4:3*h//4, w//4:3*w//4] = 1.0
    
    results = []
    
    for preset in LightingPreset:
        result = apply_studio_lighting(image, mask, preset, strength=0.8)
        
        # Добавляем подпись
        cv2.putText(result, preset.value, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        results.append(result)
        
        # Сохраняем
        output_path = f"lighting_{preset.value}.jpg"
        cv2.imwrite(output_path, result)
        print(f"✅ {preset.value}: {output_path}")
    
    # Создаём сравнительную сетку
    # 2 ряда по 4 изображения
    row1 = np.hstack(results[:4])
    row2 = np.hstack(results[4:])
    comparison = np.vstack([row1, row2])
    
    cv2.imwrite("lighting_comparison.jpg", comparison)
    print("\n✅ Сравнение: lighting_comparison.jpg")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Studio Lighting")
    parser.add_argument('--image', '-i', help='Input image path')
    parser.add_argument('--mask', '-m', help='Mask image path')
    parser.add_argument('--preset', '-p', default='cyberpunk',
                       choices=[p.value for p in LightingPreset])
    parser.add_argument('--output', '-o', default='lit_output.jpg')
    parser.add_argument('--strength', '-s', type=float, default=0.8)
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize lighting setups')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all presets')
    
    args = parser.parse_args()
    
    if args.visualize:
        # Визуализируем все схемы освещения
        setups = []
        for preset in LightingPreset:
            setup = visualize_lighting_setup(preset)
            setups.append(setup)
        
        row1 = np.hstack(setups[:4])
        row2 = np.hstack(setups[4:])
        all_setups = np.vstack([row1, row2])
        
        cv2.imwrite("lighting_setups.jpg", all_setups)
        print("✅ Схемы освещения: lighting_setups.jpg")
    
    elif args.test_all and args.image:
        test_all_presets(args.image, args.mask)
    
    elif args.image:
        image = cv2.imread(args.image)
        
        if args.mask:
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0
        else:
            h, w = image.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)
        
        preset = LightingPreset(args.preset)
        result = apply_studio_lighting(image, mask, preset, strength=args.strength)
        
        cv2.imwrite(args.output, result)
        print(f"✅ Результат: {args.output}")
    
    else:
        print("Использование:")
        print("  python studio_lighting.py --visualize  # Показать схемы")
        print("  python studio_lighting.py -i image.jpg -m mask.jpg -p cyberpunk")
        print("  python studio_lighting.py -i image.jpg --test-all")
