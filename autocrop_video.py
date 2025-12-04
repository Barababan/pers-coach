"""
АВТОМАТИЧЕСКОЕ КАДРИРОВАНИЕ ВИДЕО ТРЕНЕРА
==========================================
Анализирует видео, находит границы тела на 15 кадрах,
вычисляет оптимальный crop чтобы 80% кадров помещались.

Использование:
    python autocrop_video.py input.mp4 output.mp4
    python autocrop_video.py input.mp4 output.mp4 --samples 20 --fit 85
    python autocrop_video.py input.mp4 output.mp4 --preview  # только показать crop без обработки

После кропа загрузите output.mp4 на Google Drive и используйте в Colab.
"""

import cv2
import numpy as np
import argparse
import sys
import os


def get_person_mask(frame_bgr):
    """
    Простой метод для получения маски человека.
    Использует GrabCut + детекцию кожи - достаточно для определения bbox.
    """
    h, w = frame_bgr.shape[:2]
    
    # Метод 1: Детекция кожи в YCrCb
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    
    # Метод 2: Детекция движения/переднего плана через цвет
    # Предполагаем что человек в центре
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Создаём начальную маску - прямоугольник в центре
    mask = np.zeros((h, w), np.uint8)
    
    # GrabCut для уточнения
    rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    except:
        # Если GrabCut не сработал, используем только кожу
        mask2 = (skin_mask > 0).astype('uint8')
    
    # Комбинируем с детекцией кожи
    combined = cv2.bitwise_or(mask2 * 255, skin_mask)
    
    # Морфология для очистки
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Заполняем дыры
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(combined)
        cv2.drawContours(filled, [largest], -1, 255, -1)
        combined = filled
    
    return combined.astype(np.float32) / 255.0


def get_bbox_from_mask(mask, threshold=0.5):
    """Получить bounding box из маски"""
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0
    
    binary = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    return (x, y, x + w, y + h)


def analyze_video(video_path, num_samples=15, fit_percentage=80):
    """
    Анализирует видео и находит оптимальный crop.
    
    Args:
        video_path: путь к видео
        num_samples: сколько кадров анализировать
        fit_percentage: какой процент кадров должен помещаться
    
    Returns:
        dict с информацией о crop
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Видео: {width}x{height}, {total_frames} кадров, {fps:.1f} fps")
    print(f"🔍 Анализируем {num_samples} кадров...")
    
    # Равномерно распределённые кадры
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    bboxes = []
    sample_frames = []
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        print(f"   [{i+1}/{num_samples}] Frame {frame_idx}...", end=" ")
        
        mask = get_person_mask(frame)
        bbox = get_bbox_from_mask(mask)
        
        if bbox:
            bboxes.append(bbox)
            sample_frames.append((frame_idx, frame, mask, bbox))
            print(f"bbox: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
        else:
            print("человек не найден")
    
    cap.release()
    
    if len(bboxes) < 3:
        print("❌ Недостаточно кадров с человеком!")
        return None
    
    # === ВЫЧИСЛЯЕМ ОПТИМАЛЬНЫЙ CROP ===
    percentile_low = 100 - fit_percentage
    percentile_high = fit_percentage
    
    x_mins = [b[0] for b in bboxes]
    y_mins = [b[1] for b in bboxes]
    x_maxs = [b[2] for b in bboxes]
    y_maxs = [b[3] for b in bboxes]
    
    crop_x_min = int(np.percentile(x_mins, percentile_low))
    crop_y_min = int(np.percentile(y_mins, percentile_low))
    crop_x_max = int(np.percentile(x_maxs, percentile_high))
    crop_y_max = int(np.percentile(y_maxs, percentile_high))
    
    # Padding
    body_width = crop_x_max - crop_x_min
    body_height = crop_y_max - crop_y_min
    
    padding_x = int(body_width * 0.15)
    padding_y_top = int(body_height * 0.1)
    padding_y_bottom = int(body_height * 0.05)
    
    crop_x_min = max(0, crop_x_min - padding_x)
    crop_y_min = max(0, crop_y_min - padding_y_top)
    crop_x_max = min(width, crop_x_max + padding_x)
    crop_y_max = min(height, crop_y_max + padding_y_bottom)
    
    # Проверяем fit
    fits_count = sum(1 for bbox in bboxes 
                     if bbox[0] >= crop_x_min and bbox[1] >= crop_y_min 
                     and bbox[2] <= crop_x_max and bbox[3] <= crop_y_max)
    
    actual_fit = fits_count / len(bboxes) * 100
    
    crop_info = {
        "original_size": (width, height),
        "crop_box": (crop_x_min, crop_y_min, crop_x_max, crop_y_max),
        "crop_size": (crop_x_max - crop_x_min, crop_y_max - crop_y_min),
        "total_frames": total_frames,
        "fps": fps,
        "num_samples": len(bboxes),
        "frames_fit": fits_count,
        "fit_percentage": actual_fit,
        "sample_frames": sample_frames,
    }
    
    print(f"\n✅ Оптимальный crop:")
    print(f"   Оригинал: {width}x{height}")
    print(f"   Crop box: ({crop_x_min}, {crop_y_min}) - ({crop_x_max}, {crop_y_max})")
    print(f"   Новый размер: {crop_x_max - crop_x_min}x{crop_y_max - crop_y_min}")
    print(f"   Кадров помещается: {fits_count}/{len(bboxes)} ({actual_fit:.0f}%)")
    
    return crop_info


def show_preview(crop_info):
    """Показать превью кадрирования"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib не установлен, превью недоступно")
        return
    
    sample_frames = crop_info["sample_frames"]
    crop_box = crop_info["crop_box"]
    
    num_show = min(6, len(sample_frames))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, ax in enumerate(axes.flat):
        if idx >= num_show:
            ax.axis('off')
            continue
        
        frame_idx, frame, mask, bbox = sample_frames[idx * len(sample_frames) // num_show]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Person bbox (green)
        from matplotlib.patches import Rectangle
        rect_person = Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            fill=False, edgecolor='lime', linewidth=2
        )
        ax.add_patch(rect_person)
        
        # Crop area (red dashed)
        rect_crop = Rectangle(
            (crop_box[0], crop_box[1]), 
            crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
            fill=False, edgecolor='red', linewidth=2, linestyle='--'
        )
        ax.add_patch(rect_crop)
        
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
    
    plt.suptitle(
        f'AUTO CROP: {crop_info["crop_size"][0]}x{crop_info["crop_size"][1]} | '
        f'{crop_info["fit_percentage"]:.0f}% fit',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def crop_video(input_path, output_path, crop_info):
    """Кропает видео по найденным границам"""
    crop_box = crop_info["crop_box"]
    x1, y1, x2, y2 = crop_box
    
    cap = cv2.VideoCapture(input_path)
    fps = crop_info["fps"]
    new_width = x2 - x1
    new_height = y2 - y1
    
    # Используем mp4v кодек
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    total_frames = crop_info["total_frames"]
    print(f"\n🎬 Кропаем видео: {new_width}x{new_height}")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop
        cropped = frame[y1:y2, x1:x2]
        out.write(cropped)
        
        frame_num += 1
        if frame_num % 100 == 0:
            progress = frame_num / total_frames * 100
            print(f"   {frame_num}/{total_frames} ({progress:.0f}%)")
    
    cap.release()
    out.release()
    
    # Получаем размер файла
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✅ Готово: {output_path}")
    print(f"   Размер: {file_size:.1f} MB")
    print(f"   Разрешение: {new_width}x{new_height}")
    print(f"\n📤 Загрузите файл на Google Drive и используйте в Colab!")


def main():
    parser = argparse.ArgumentParser(
        description='Автоматическое кадрирование видео тренера',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python autocrop_video.py squat.mp4 squat_cropped.mp4
  python autocrop_video.py squat.mp4 squat_cropped.mp4 --samples 20
  python autocrop_video.py squat.mp4 squat_cropped.mp4 --fit 85
  python autocrop_video.py squat.mp4 --preview
        """
    )
    
    parser.add_argument('input', help='Входное видео')
    parser.add_argument('output', nargs='?', help='Выходное видео (опционально для --preview)')
    parser.add_argument('--samples', type=int, default=15, help='Количество кадров для анализа (default: 15)')
    parser.add_argument('--fit', type=int, default=80, help='Процент кадров которые должны помещаться (default: 80)')
    parser.add_argument('--preview', action='store_true', help='Только показать превью, не кропать')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Файл не найден: {args.input}")
        sys.exit(1)
    
    if not args.preview and not args.output:
        print("❌ Укажите выходной файл или используйте --preview")
        sys.exit(1)
    
    print("=" * 50)
    print("🎬 АВТОМАТИЧЕСКОЕ КАДРИРОВАНИЕ ВИДЕО")
    print("=" * 50)
    

    
    # Анализируем видео
    crop_info = analyze_video(args.input, args.samples, args.fit)
    
    if not crop_info:
        sys.exit(1)
    
    if args.preview:
        show_preview(crop_info)
    else:
        crop_video(args.input, args.output, crop_info)


if __name__ == "__main__":
    main()
