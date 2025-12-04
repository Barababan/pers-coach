"""
SMART MIRROR CONFIGURATION
==========================
Финальные настройки обработки видео тренера для Smart Mirror.
Версия: V11 Manual (лучший натуральный результат)

Использование:
    from smart_mirror_config import SMART_MIRROR_CONFIG, process_trainer_frame
    
    # Обработать кадр
    trainer_rgba = process_trainer_frame(frame_bgr, person_mask)
"""

import cv2
import numpy as np

# ============================================================
# 🏆 V11 MANUAL - ЛУЧШАЯ КОНФИГУРАЦИЯ
# ============================================================

SMART_MIRROR_CONFIG = {
    # Метаданные
    "version": "V11_manual",
    "description": "Мягкая коррекция яркости/контраста - натуральный вид без пластиковости",
    
    # === Основные параметры коррекции ===
    "brightness_boost": 0.08,      # +8% яркость (поверх V6)
    "contrast": 1.12,              # +12% контраст
    
    # === V6 Base: LAB коррекция ===
    "lab_lift": 52,                # Подъём яркости в L канале
    
    # === Skin Enhancement ===
    "skin_saturation_boost": 15,   # Насыщенность кожи в HSV
    "skin_detection": {
        "ycrcb_lower": (0, 133, 77),   # Нижняя граница YCrCb
        "ycrcb_upper": (255, 173, 127), # Верхняя граница YCrCb
        "blur_kernel": 15               # Размытие маски кожи
    },
    
    # === Цветокоррекция (теплота) ===
    "warmth": {
        "red_boost": 1.03,         # Множитель красного канала
        "blue_reduce": 0.97        # Множитель синего канала
    },
    
    # === Маска (MODNet) ===
    "mask_blur_kernel": 31,        # Размытие границ маски человека
}

# ============================================================
# ФУНКЦИИ ОБРАБОТКИ
# ============================================================

def process_trainer_frame(image_bgr, person_mask, config=None):
    """
    Обработать кадр тренера с настройками V11.
    
    Args:
        image_bgr: Входное изображение в BGR формате (numpy array)
        person_mask: Маска человека (numpy array, float 0-1 или uint8 0-255)
        config: Словарь с настройками (по умолчанию SMART_MIRROR_CONFIG)
    
    Returns:
        RGBA изображение (numpy array) с прозрачным фоном
    """
    if config is None:
        config = SMART_MIRROR_CONFIG
    
    # Нормализуем маску
    if person_mask.max() > 1:
        person_mask = person_mask.astype(np.float32) / 255.0
    
    h, w = image_bgr.shape[:2]
    
    # 1. LAB коррекция яркости
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    blur_k = config["mask_blur_kernel"]
    mask_soft = cv2.GaussianBlur(person_mask.astype(np.float32), (blur_k, blur_k), 0)
    
    L_new = L + config["lab_lift"] * mask_soft
    L_new = np.clip(L_new, 0, 255)
    
    lab_new = np.stack([L_new, A, B], axis=-1).astype(np.uint8)
    enhanced = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
    
    # 2. Skin enhancement
    ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
    skin_det = config["skin_detection"]
    skin_mask = cv2.inRange(ycrcb, skin_det["ycrcb_lower"], skin_det["ycrcb_upper"])
    skin_blur = skin_det["blur_kernel"]
    skin_mask = cv2.GaussianBlur(skin_mask, (skin_blur, skin_blur), 0)
    skin_mask = (skin_mask.astype(np.float32) / 255.0) * person_mask
    
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] + config["skin_saturation_boost"] * skin_mask
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 3. V11 boost (яркость + контраст)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_f = enhanced_rgb.astype(np.float32) / 255.0
    img_f = img_f + config["brightness_boost"]
    img_f = (img_f - 0.5) * config["contrast"] + 0.5
    img_f = np.clip(img_f, 0, 1)
    
    # 4. Warmth (теплота)
    warmth = config["warmth"]
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * warmth["red_boost"], 0, 1)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * warmth["blue_reduce"], 0, 1)
    
    result_rgb = (img_f * 255).astype(np.uint8)
    
    # 5. Создаём RGBA (прозрачный фон)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_rgba = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = (person_mask * 255).astype(np.uint8)
    
    return result_rgba


def create_background(h, w, bg_type="transparent"):
    """
    Создать фон для Smart Mirror.
    
    Args:
        h, w: Размеры изображения
        bg_type: Тип фона:
            - "transparent": Прозрачный (для наложения на видео юзера) - РЕКОМЕНДУЕТСЯ
            - "dark": Тёмный однотонный (RGB 45,45,45)
            - "gray": Нейтральный серый (RGB 128,128,128)
            - "gradient_v": Вертикальный градиент (светлый сверху)
            - "radial": Радиальный градиент (светлый в центре)
            - "gym": Имитация спортзала (стена + пол)
            - "mirror": Эффект зеркала
            - "blue_tint": Холодный синеватый оттенок
    
    Returns:
        numpy array (h, w, 3) или None для прозрачного
    """
    if bg_type == "transparent":
        return None
    
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    if bg_type == "dark":
        bg[:, :] = [45, 45, 45]
        
    elif bg_type == "gray":
        bg[:, :] = [128, 128, 128]
        
    elif bg_type == "gradient_v":
        for i in range(h):
            val = int(160 - (i / h) * 80)
            bg[i, :] = [val, val, val]
            
    elif bg_type == "radial":
        center_y, center_x = h // 2, w // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                val = int(180 - (dist / max_dist) * 100)
                bg[y, x] = [val, val, val]
                
    elif bg_type == "gym":
        wall_height = int(h * 0.4)
        bg[:wall_height, :] = [180, 180, 180]
        for i in range(wall_height, h):
            progress = (i - wall_height) / (h - wall_height)
            val = int(120 - progress * 40)
            bg[i, :] = [val, val, val]
            
    elif bg_type == "mirror":
        bg[:, :] = [100, 100, 100]
        floor_start = int(h * 0.85)
        for i in range(floor_start, h):
            progress = (i - floor_start) / (h - floor_start)
            val = int(100 - progress * 30)
            bg[i, :] = [val, val, val]
            
    elif bg_type == "blue_tint":
        for i in range(h):
            val = int(140 - (i / h) * 50)
            bg[i, :] = [val + 10, val + 5, val]
    
    return bg


def composite_trainer_on_background(trainer_rgba, background=None):
    """
    Наложить тренера на фон.
    
    Args:
        trainer_rgba: RGBA изображение тренера
        background: Фон (numpy array) или None для прозрачного
    
    Returns:
        RGB изображение (если есть фон) или RGBA (если прозрачный)
    """
    if background is None:
        return trainer_rgba
    
    alpha = trainer_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = trainer_rgba[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    
    result = fg * alpha + bg * (1 - alpha)
    return result.astype(np.uint8)


# ============================================================
# COLAB RESTORATION CODE
# ============================================================
# Код для восстановления в Google Colab:

COLAB_SETUP_CODE = '''
# === УСТАНОВКА В COLAB ===
!pip install -q torch torchvision opencv-python-headless

# Клонируем MODNet для сегментации
!git clone https://github.com/ZHKKKe/MODNet.git
!pip install -q gdown
import gdown
import os
os.makedirs('MODNet/pretrained', exist_ok=True)
gdown.download(
    'https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz',
    'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt',
    quiet=False
)

# Загрузка MODNet
import sys
sys.path.insert(0, 'MODNet')
from src.models.modnet import MODNet
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)
modnet.load_state_dict(torch.load('MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', map_location='cpu'))
modnet = modnet.module.to(device)
modnet.eval()

def get_person_mask(image_bgr):
    """Получить маску человека через MODNet"""
    h, w = image_bgr.shape[:2]
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
    return matte[0, 0].cpu().numpy()

print("✅ MODNet готов!")
'''

if __name__ == "__main__":
    print("Smart Mirror Configuration")
    print("=" * 40)
    print(f"Version: {SMART_MIRROR_CONFIG['version']}")
    print(f"Description: {SMART_MIRROR_CONFIG['description']}")
    print()
    print("Settings:")
    print(f"  Brightness: +{SMART_MIRROR_CONFIG['brightness_boost']*100:.0f}%")
    print(f"  Contrast: +{(SMART_MIRROR_CONFIG['contrast']-1)*100:.0f}%")
    print(f"  LAB lift: {SMART_MIRROR_CONFIG['lab_lift']}")
    print(f"  Skin saturation: +{SMART_MIRROR_CONFIG['skin_saturation_boost']}")
    print()
    print("Available backgrounds:")
    print("  - transparent (recommended)")
    print("  - dark, gray, gradient_v, radial, gym, mirror, blue_tint")
