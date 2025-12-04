# 🚀 Lightning.AI Setup Guide

## Полная инструкция по настройке GPU-сервера для pers.coach

---

## 1. Регистрация и создание Studio

### 1.1 Регистрация
1. Перейди на https://lightning.ai/
2. Нажми **"Start free"**
3. Войди через **Google** (mshagiev@gmail.com)
4. Потребуется **номер телефона** для верификации

### 1.2 Free Tier
- **15 кредитов/месяц** (~80 GPU часов на spot)
- **T4 GPU**: $0.19/час
- **L4 GPU**: $0.48/час

### 1.3 Создание Studio
1. Нажми **"New Studio"** или **"+"**
2. Выбери **GPU: T4** (16 GB VRAM, достаточно для наших задач)
3. Дай название: `smart-mirror-processing`

---

## 2. Подключение к Studio

### 2.1 SSH Setup (один раз)

В веб-интерфейсе Lightning.AI нажми **SSH** и скопируй команду:

```bash
curl -s "https://lightning.ai/setup/ssh?t=YOUR_TOKEN&s=YOUR_STUDIO_ID" | bash
```

Это создаст:
- SSH ключ: `~/.ssh/lightning_rsa`
- Конфиг: `~/.ssh/config`

### 2.2 Подключение через терминал

```bash
ssh s_YOUR_STUDIO_ID@ssh.lightning.ai
```

### 2.3 Подключение через VS Code

1. В Lightning.AI нажми **"Connect"** → **"Connect local VSCode IDE"**
2. VS Code откроется с расширением Remote-SSH
3. Терминал в VS Code = терминал на GPU-сервере

---

## 3. Загрузка файлов на сервер

### 3.1 Через SCP

```bash
# Загрузка одного файла
scp /path/to/local/file.py s_YOUR_STUDIO_ID@ssh.lightning.ai:/teamspace/studios/this_studio/

# Загрузка папки
scp -r /path/to/local/folder s_YOUR_STUDIO_ID@ssh.lightning.ai:/teamspace/studios/this_studio/
```

### 3.2 Рабочая директория

На Lightning.AI все файлы в:
```
/teamspace/studios/this_studio/
```

---

## 4. Задача 1: Обработка видео тренера (MODNet + V11)

### 4.1 Файлы для загрузки

```bash
scp squat_cropped.mp4 s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/
scp lightning_trainer_pipeline.ipynb s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/
```

### 4.2 Установка зависимостей

```bash
pip install torch torchvision opencv-python-headless Pillow tqdm
apt-get update && apt-get install -y ffmpeg

# Клонируем MODNet
git clone https://github.com/ZHKKKe/MODNet.git

# Скачиваем веса MODNet
mkdir -p MODNet/pretrained
wget -O MODNet/pretrained/modnet_webcam_portrait_matting.ckpt \
    "https://drive.google.com/uc?export=download&id=1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX"
```

### 4.3 Запуск обработки

Открой `lightning_trainer_pipeline.ipynb` и запусти все ячейки.

Или создай скрипт:

```python
# process_trainer_video.py
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Пути
PROJECT_DIR = "/teamspace/studios/this_studio"
INPUT_VIDEO = os.path.join(PROJECT_DIR, "squat_cropped.mp4")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
OUTPUT_PNG_DIR = os.path.join(OUTPUT_DIR, "trainer_frames")

sys.path.insert(0, os.path.join(PROJECT_DIR, "MODNet/src"))
from models.modnet import MODNet
from torchvision import transforms

# V11 настройки
V11_CONFIG = {
    'brightness_boost': 0.08,
    'contrast': 1.12,
    'lab_lift': 52,
    'warmth': {'red_boost': 1.03, 'blue_reduce': 0.97}
}

# ... (полный код в lightning_trainer_pipeline.ipynb)
```

### 4.4 Создание WebM с альфа

```bash
ffmpeg -y -framerate 30 \
    -i output/trainer_frames/%05d.png \
    -c:v libvpx-vp9 \
    -pix_fmt yuva420p \
    -b:v 2M \
    -auto-alt-ref 0 \
    output/trainer_transparent.webm
```

### 4.5 Выходные файлы

```
output/
├── trainer_transparent.webm  # ~62 MB, VP9 + alpha
└── trainer_frames/           # PNG с прозрачностью
```

---

## 5. Задача 2: Извлечение 3D позы (SAM 3D Body)

### 5.1 Файлы для загрузки

```bash
scp squat_cropped.mp4 s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/
scp extract_3d_pose.py s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/
```

### 5.2 Установка зависимостей

```bash
# SAM 3D Body зависимости
pip install pytorch-lightning pyrender opencv-python yacs scikit-image \
    einops timm dill pandas rich hydra-core pyrootutils networkx==3.2.1 \
    roma joblib huggingface_hub

# Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# Клонируем SAM 3D Body
git clone https://github.com/facebookresearch/sam-3d-body.git
```

### 5.3 Скачивание чекпоинтов

⚠️ Нужен доступ к HuggingFace: https://huggingface.co/facebook/sam-3d-body-dinov3

```bash
# Логин в HuggingFace
huggingface-cli login
# Вставь токен

# Скачиваем чекпоинты (~800 MB)
huggingface-cli download facebook/sam-3d-body-dinov3 \
    --local-dir checkpoints/sam-3d-body-dinov3
```

### 5.4 Запуск извлечения 3D позы

```bash
python extract_3d_pose.py \
    --video squat_cropped.mp4 \
    --output output/poses \
    --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --skip 2 \
    --device cuda
```

**Параметры:**
- `--skip 2` — каждый 2-й кадр (15 fps эффективных, быстрее)
- `--skip 1` — все кадры (30 fps, медленнее)

**Время:** ~15-30 минут на T4 GPU

### 5.5 Выходные файлы

```
output/poses/
├── trainer_poses.json  # 50 MB - для браузера (сравнение поз)
└── trainer_poses.npz   # 8.4 MB - для 3D визуализации
```

**Структура JSON:**
```json
{
  "version": "1.0",
  "model": "SAM-3D-Body",
  "fps": 15,
  "total_frames": 2897,
  "frame_indices": [0, 2, 4, ...],
  "poses": [
    {
      "joints_3d": [[x, y, z], ...],
      "keypoints_2d": [[x, y], ...],
      "global_rot": [rx, ry, rz],
      "body_pose": [...]
    },
    ...
  ]
}
```

---

## 6. Скачивание результатов

```bash
# 3D позы
scp s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/output/poses/trainer_poses.json ./output/poses/
scp s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/output/poses/trainer_poses.npz ./output/poses/

# Прозрачное видео
scp s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/output/trainer_transparent.webm ./output/
```

---

## 7. Быстрый старт (команды одной строкой)

### Полный пайплайн для нового видео:

```bash
# 1. Загрузить видео
scp my_video.mp4 s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/

# 2. На сервере: обработка MODNet + V11
ssh s_STUDIO@ssh.lightning.ai "cd /teamspace/studios/this_studio && python process_trainer_video.py --input my_video.mp4"

# 3. На сервере: 3D поза
ssh s_STUDIO@ssh.lightning.ai "cd /teamspace/studios/this_studio && python extract_3d_pose.py --video my_video.mp4 --output output/poses --skip 2 --device cuda"

# 4. Скачать результаты
scp s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/output/trainer_transparent.webm ./
scp s_STUDIO@ssh.lightning.ai:/teamspace/studios/this_studio/output/poses/*.json ./output/poses/
```

---

## 8. Текущий Studio ID

```
Studio ID: s_01kbm748xt78mk87pbz5p7c5dm
SSH: ssh s_01kbm748xt78mk87pbz5p7c5dm@ssh.lightning.ai
```

---

## 9. Стоимость

| Задача | GPU | Время | Стоимость |
|--------|-----|-------|-----------|
| MODNet + V11 (5800 кадров) | T4 | ~10 мин | ~$0.03 |
| SAM 3D Body (2900 кадров) | T4 | ~20 мин | ~$0.06 |
| **Итого** | | ~30 мин | **~$0.10** |

Free tier (15 кредитов) хватит на ~150 таких обработок.
