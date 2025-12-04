# Smart Mirror Pipeline - Полная документация

## Обзор

Пайплайн для обработки видео тренера: удаление фона, цветокоррекция V11, cyclorama фон.

---

## 1. Локальный скрипт: Автокадрирование видео

### Файл: `autocrop_video.py`

**Назначение:** Автоматически обрезает видео по границам тела тренера.

**Использование:**
```bash
# Базовое использование
python3 autocrop_video.py squat.mp4 squat_cropped.mp4

# С параметрами
python3 autocrop_video.py squat.mp4 squat_cropped.mp4 --samples 20 --fit 85

# Только превью (без обработки)
python3 autocrop_video.py squat.mp4 --preview
```

**Параметры:**
- `--samples N` — количество кадров для анализа (default: 15)
- `--fit N` — процент кадров которые должны помещаться (default: 80)
- `--preview` — только показать bbox, не обрабатывать

**Как работает:**
1. Берёт N случайных кадров из видео
2. На каждом находит человека через GrabCut + детекция кожи
3. Собирает bounding boxes всех кадров
4. Вычисляет crop который вмещает заданный % кадров
5. Обрезает видео через ffmpeg

**Зависимости:** opencv-python, numpy, ffmpeg (системный)

---

## 2. Google Colab: Обработка видео

### Подготовка видео

1. Локально запустить autocrop:
   ```bash
   python3 autocrop_video.py squat.mp4 squat_cropped.mp4 --fit 85
   ```

2. Загрузить `squat_cropped.mp4` на Google Drive

3. Получить ссылку: ПКМ → "Получить ссылку" → "Все у кого есть ссылка"

4. Извлечь FILE_ID из ссылки:
   ```
   https://drive.google.com/file/d/XXXXXXXXXXXXX/view
                                    ↑ это FILE_ID
   ```

### Порядок запуска ячеек в Colab

#### Ячейка 1: Проверка GPU
```python
!nvidia-smi
```

#### Ячейка 2: Установка зависимостей
```python
%cd /content
!pip install onnxruntime-gpu gdown
!git clone https://github.com/ZHKKKe/MODNet.git
# Веса скачиваются отдельно через gdown
```

#### Ячейка 3: Загрузка видео с Google Drive
```python
import gdown
import os

file_id = "ВАШ_FILE_ID"  # <-- заменить!
output_video = "/content/squat_cropped.mp4"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output_video, quiet=False)

if os.path.exists(output_video):
    print(f"✅ Видео загружено")
    !ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 {output_video}
```

#### Ячейка 4: Инициализация MODNet
```python
import torch
import sys
sys.path.insert(0, '/content/MODNet/src')
from models.modnet import MODNet
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)
modnet.load_state_dict(torch.load('MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', map_location='cpu'))
modnet = modnet.module.to(device)
modnet.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(f"✅ MODNet загружен на {device}")
```

#### Ячейка 5: V11 функции обработки
```python
import cv2
import numpy as np
from PIL import Image

# V11 Настройки (проверенные, естественный вид)
V11_CONFIG = {
    'brightness_boost': 0.08,      # +8%
    'contrast': 1.12,              # +12%
    'lab_lift': 52,                # Осветление теней в LAB
    'skin_saturation_boost': 15,   # Насыщенность кожи
    'warmth': {'red_boost': 1.03, 'blue_reduce': 0.97}
}

def process_frame_v11(frame_bgr, mask=None):
    """Обработка кадра с V11 настройками"""
    img = frame_bgr.astype(np.float32) / 255.0
    
    # 1. Яркость
    img = img + V11_CONFIG['brightness_boost']
    
    # 2. Контраст
    img = (img - 0.5) * V11_CONFIG['contrast'] + 0.5
    
    # 3. LAB lift (осветление теней)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]
    dark_mask = L < V11_CONFIG['lab_lift']
    lift_amount = (V11_CONFIG['lab_lift'] - L) * 0.3
    lab[:,:,0] = np.where(dark_mask, L + lift_amount, L)
    lab[:,:,0] = np.clip(lab[:,:,0], 0, 255)
    img_uint8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    img = img_uint8.astype(np.float32) / 255.0
    
    # 4. Теплота
    img[:,:,2] = img[:,:,2] * V11_CONFIG['warmth']['red_boost']    # R
    img[:,:,0] = img[:,:,0] * V11_CONFIG['warmth']['blue_reduce']  # B
    
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def create_cyclorama_background(width, height, style="light"):
    """
    Создаёт cyclorama фон с 3D глубиной
    
    style: "dark" или "light"
    """
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    
    if style == "light":
        # Светлая студия (как Apple)
        wall_color = (180, 180, 185)   # Светло-серая стена
        floor_color = (160, 160, 165)  # Чуть темнее пол
    else:
        # Тёмная студия
        wall_color = (45, 42, 40)
        floor_color = (35, 32, 30)
    
    horizon = int(height * 0.7)
    transition_height = int(height * 0.15)
    
    # Стена с градиентом
    for y in range(horizon):
        t = y / horizon
        factor = 1.0 - t * 0.08  # Небольшое затемнение к горизонту
        color = tuple(int(c * factor) for c in wall_color)
        bg[y, :] = color
    
    # S-образный переход (cyclorama curve)
    for y in range(horizon, min(horizon + transition_height, height)):
        t = (y - horizon) / transition_height
        s = t * t * (3 - 2 * t)  # smoothstep
        color = tuple(int(wall_color[i] * (1 - s) + floor_color[i] * s) for i in range(3))
        bg[y, :] = color
    
    # Пол с перспективным затемнением
    for y in range(horizon + transition_height, height):
        t = (y - horizon - transition_height) / (height - horizon - transition_height + 1)
        factor = 1.0 - t * 0.15
        color = tuple(int(c * factor) for c in floor_color)
        bg[y, :] = color
    
    # Мягкое размытие
    bg = cv2.bilateralFilter(bg, 9, 75, 75)
    
    return bg


def composite_on_background(foreground_bgr, mask, background):
    """Композитинг на фон с учётом маски"""
    if mask.ndim == 2:
        mask_3ch = np.stack([mask] * 3, axis=-1)
    else:
        mask_3ch = mask
    
    fg = foreground_bgr.astype(np.float32)
    bg = cv2.resize(background, (foreground_bgr.shape[1], foreground_bgr.shape[0])).astype(np.float32)
    mask_f = mask_3ch.astype(np.float32)
    if mask_f.max() > 1:
        mask_f = mask_f / 255.0
    
    result = fg * mask_f + bg * (1 - mask_f)
    return np.clip(result, 0, 255).astype(np.uint8)
```

#### Ячейка 6: Обработка видео
```python
input_video = "/content/squat_cropped.mp4"
output_video = "/content/squat_processed.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 Видео: {width}x{height}, {fps} fps, {total_frames} кадров")

# Паддинг для MODNet (требует размер кратный 32)
def pad_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h, pad_w = new_h - h, new_w - w
    if len(img.shape) == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect'), (h, w)
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect'), (h, w)

# СВЕТЛАЯ cyclorama!
background = create_cyclorama_background(width, height, style="light")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

max_frames = None  # или 100 для теста

from tqdm import tqdm
frame_count = 0

with tqdm(total=min(total_frames, max_frames or total_frames)) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # MODNet маска
        frame_padded, orig_size = pad_to_multiple(frame, 32)
        img_pil = Image.fromarray(cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB))
        tensor = transform(img_pil).unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, _, mask_tensor = modnet(tensor, True)
        
        mask = mask_tensor[0, 0].cpu().numpy()[:orig_size[0], :orig_size[1]]
        
        # V11 обработка
        processed = process_frame_v11(frame, mask)
        
        # Композитинг
        final = composite_on_background(processed, mask, background)
        
        out.write(final)
        frame_count += 1
        pbar.update(1)

cap.release()
out.release()
print(f"✅ Готово! {frame_count} кадров → {output_video}")
```

#### Ячейка 7: Превью и скачивание
```python
import matplotlib.pyplot as plt

# Превью
cap = cv2.VideoCapture(output_video)
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for i, idx in enumerate([0, total_frames//4, total_frames//2, 3*total_frames//4]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, frame_count-1))
    ret, frame = cap.read()
    if ret:
        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Кадр {idx}')
        axes[i].axis('off')
cap.release()
plt.show()

# Конвертация в H.264 и скачивание
!ffmpeg -y -i {output_video} -c:v libx264 -preset fast -crf 23 /content/squat_final.mp4

from google.colab import files
files.download('/content/squat_final.mp4')
```

---

## 3. V11 Настройки (финальные)

Эти настройки дают **естественный** вид без "пластикового" эффекта:

| Параметр | Значение | Описание |
|----------|----------|----------|
| brightness_boost | 0.08 | +8% яркости |
| contrast | 1.12 | +12% контраста |
| lab_lift | 52 | Осветление теней < 52 в LAB |
| skin_saturation_boost | 15 | Насыщенность кожи |
| red_boost | 1.03 | Тёплый оттенок (красный +3%) |
| blue_reduce | 0.97 | Тёплый оттенок (синий -3%) |

---

## 4. Cyclorama фон

**Светлая студия (style="light"):**
- Стена: RGB(180, 180, 185) — светло-серая
- Пол: RGB(160, 160, 165) — чуть темнее

**Тёмная студия (style="dark"):**
- Стена: RGB(45, 42, 40)
- Пол: RGB(35, 32, 30)

**Эффекты:**
- Горизонт на 70% высоты
- S-curve переход (smoothstep) между стеной и полом
- Градиентное затемнение к краям
- Bilateral filter для мягкости

---

## 5. Важные моменты

### MODNet требует паддинг
Размер изображения должен быть кратен 32. Используйте `pad_to_multiple()`.

### Autocrop 80% vs 90%
- `--fit 80` — компактный crop, могут обрезаться крайние позы
- `--fit 90` — безопаснее, больше запаса

### Скорость обработки
- Tesla T4: ~10 fps (5000 кадров ≈ 8-10 минут)
- Без GPU: очень медленно, не рекомендуется

---

## 6. Файлы проекта

```
/Users/user/Documents/pers2/
├── autocrop_video.py          # Локальный скрипт кадрирования
├── smart_mirror_config.py     # V11 конфиг + функции
├── colab_relighting_pipeline.ipynb  # Основной notebook
├── squat.mp4                  # Исходное видео
├── squat_cropped.mp4          # После autocrop
└── SMART_MIRROR_PIPELINE.md   # Эта документация
```

---

## 7. Quick Start

```bash
# 1. Локально: обрезка
python3 autocrop_video.py squat.mp4 squat_cropped.mp4 --fit 85

# 2. Загрузить squat_cropped.mp4 на Google Drive

# 3. В Colab: запустить ячейки 1-7

# 4. Скачать squat_final.mp4
```
