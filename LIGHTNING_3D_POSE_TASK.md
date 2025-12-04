# 🚀 Lightning.AI - Полное описание проекта Smart Mirror

## Проект: pers.coach — Shadow Training Platform

**Цель:** Создать платформу где клиент повторяет движения тренера в реальном времени через вебкамеру, с визуальным фидбеком совпадения поз.

---

## 📦 Что нужно обработать на GPU

### Входные данные:
- `squat_cropped.mp4` — видео тренера (602x722, 30fps, 5793 кадра, ~3 мин)

### Выходные данные:

| Файл | Назначение | Формат |
|------|------------|--------|
| `trainer_transparent.webm` | Видео для веб-страницы | VP9 + alpha (62 MB) |
| `trainer_poses.json` | 3D позы для сравнения | JSON с landmarks |
| `trainer_poses.npz` | Данные для 3D человечка | NPZ (vertices, faces) |

---

## 🎯 Задача 1: Удаление фона + Color Correction (✅ ГОТОВО)

**Статус:** Выполнено локально на CPU (44 мин), результат в `output/trainer_transparent.webm`

**Pipeline:**
1. MODNet — удаление фона
2. V11 Color Correction — студийное качество
3. FFmpeg — WebM с альфа-каналом

**V11 настройки:**
```python
brightness_boost = 0.08
contrast = 1.12
lab_lift = 52
warmth = {red_boost: 1.03, blue_reduce: 0.97}
```

---

## 🦴 Задача 2: Извлечение 3D позы (SAM 3D Body)

**Статус:** ⏳ Нужно выполнить на GPU

**Цель:** Извлечь 3D позу тренера из каждого кадра для:
1. Сравнения с позой юзера (MediaPipe в браузере)
2. Отображения 3D человечка тренера
3. Наложения скелетов тренера и юзера

---

## 🔧 Установка на Lightning.AI

### Шаг 1: Зависимости

```bash
# SAM 3D Body зависимости
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core pyrootutils networkx==3.2.1 roma joblib huggingface_hub

# Detectron2 (для детекции человека)
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Проверка GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Шаг 2: Клонирование SAM 3D Body

```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
```

### Шаг 3: Чекпоинты (⚠️ нужен доступ HuggingFace)

```bash
# Логин в HuggingFace
huggingface-cli login

# Скачиваем чекпоинты (~800 MB)
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

**Альтернатива:** Загрузить чекпоинты с локальной машины (уже скачаны в `/Users/user/Documents/pers2/checkpoints/`)

### Шаг 4: Загрузка файлов

Загрузи с локальной машины:
- `squat_cropped.mp4` — видео тренера
- `extract_3d_pose.py` — скрипт обработки

### Шаг 5: Запуск

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
- `--skip 2` — каждый 2-й кадр (быстрее, 15 fps эффективных)
- `--skip 1` — все кадры (полное качество, медленнее)

**Время:** ~15-30 мин на T4 GPU

---

## 📊 Формат выходных данных

### trainer_poses.json (для браузера)
```json
{
  "fps": 15,
  "width": 602,
  "height": 722,
  "total_frames": 2896,
  "poses": [
    {
      "joints_3d": [[x, y, z], ...],
      "keypoints_2d": [[x, y], ...],
      "global_rot": [rx, ry, rz]
    },
    ...
  ]
}
```

### trainer_poses.npz (для 3D визуализации)
```python
{
  "fps": 15,
  "joints_3d": np.array([frames, joints, 3]),
  "vertices": np.array([frames, vertices, 3]),
  "faces": np.array([faces, 3])  # Mesh topology
}
```

---

## 🖥️ Использование в браузере

### Сравнение поз
```javascript
// Загружаем JSON с позами тренера
const trainerPoses = await fetch('trainer_poses.json').then(r => r.json());

// Получаем позу юзера через MediaPipe
const userPose = await poseLandmarker.detect(webcamFrame);

// Сравниваем
const matchScore = comparePoses(
    trainerPoses.poses[currentFrame].joints_3d,
    userPose.worldLandmarks
);
```

### 3D визуализация (Three.js)
```javascript
// Загружаем NPZ (через jszip + np-loader)
const data = await loadNPZ('trainer_poses.npz');

// Создаём меш
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices[frame], 3));
geometry.setIndex(data.faces.flat());

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);
```

---

## 🔄 Альтернатива: MediaPipe Pose (если нет SAM 3D Body)

```python
import mediapipe as mp
import cv2
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Максимальное качество
    enable_segmentation=False,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture('squat_cropped.mp4')
poses = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_world_landmarks:
        landmarks = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z}
            for lm in results.pose_world_landmarks.landmark
        ]
        poses.append(landmarks)

with open('trainer_poses_mediapipe.json', 'w') as f:
    json.dump({'poses': poses}, f)
```

**Плюсы MediaPipe:**
- Не требует доступа к HuggingFace
- Быстрее (~3-5 мин на GPU)
- Совместимо с браузерным MediaPipe

**Минусы:**
- Нет полного меша (только скелет)
- Менее точно для сложных поз

---

## 📁 Итоговые файлы для скачивания

После выполнения обеих задач скачай:

```
output/
├── trainer_transparent.webm    # 62 MB - видео с альфа
├── trainer_frames/             # ~2 GB - PNG кадры (backup)
└── poses/
    ├── trainer_poses.json      # ~10 MB - для браузера
    └── trainer_poses.npz       # ~50 MB - для 3D
```

