#!/usr/bin/env python3
"""
Extract 3D Pose from Trainer Video using SAM 3D Body

Входные данные:
- Видео тренера (squat_cropped.mp4)

Выходные данные:
- JSON с 3D landmarks для каждого кадра (совместимо с MediaPipe)
- NPZ с полными данными позы для визуализации 3D человечка

Для запуска на Lightning.AI с GPU
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch

# Добавляем путь к SAM 3D Body
SAM3D_PATH = os.path.join(os.path.dirname(__file__), "sam-3d-body")
sys.path.insert(0, SAM3D_PATH)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


# Маппинг SAM 3D Body joints к MediaPipe Pose landmarks (33 точки)
# SAM 3D Body использует MHR (Momentum Human Rig) формат
# MediaPipe использует 33 landmarks

MEDIAPIPE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Основные суставы для сравнения поз (совпадают с MediaPipe)
POSE_JOINTS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}


def setup_estimator(checkpoint_path, mhr_path, device='cuda'):
    """Инициализация SAM 3D Body estimator"""
    print(f"🔧 Loading SAM 3D Body on {device}...")
    
    model, model_cfg = load_sam_3d_body(
        checkpoint_path, 
        device=torch.device(device),
        mhr_path=mhr_path
    )
    
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,  # Используем полный кадр (тренер уже в центре)
        human_segmentor=None,
        fov_estimator=None,
    )
    
    print("✅ SAM 3D Body loaded!")
    return estimator


def sam3d_to_mediapipe_format(sam3d_output):
    """
    Конвертирует выход SAM 3D Body в формат MediaPipe World Landmarks
    
    SAM 3D Body output содержит:
    - pred_keypoints_3d: 3D координаты в метрах
    - pred_keypoints_2d: 2D проекции на изображение
    - pred_joint_coords: координаты суставов MHR rig
    """
    
    # Используем pred_joint_coords - это координаты суставов скелета
    joints_3d = sam3d_output.get('pred_joint_coords')
    
    if joints_3d is None:
        # Fallback на pred_keypoints_3d
        joints_3d = sam3d_output.get('pred_keypoints_3d')
    
    if joints_3d is None:
        return None
    
    # Нормализуем координаты (центрируем относительно hip)
    # MediaPipe world landmarks центрированы относительно hip center
    
    # SAM 3D Body имеет другую топологию скелета, 
    # нужно маппить на MediaPipe 33 landmarks
    
    # Для упрощения возвращаем основные суставы тела
    # которые можно использовать для сравнения поз
    
    result = {
        'joints_3d': joints_3d.tolist() if isinstance(joints_3d, np.ndarray) else joints_3d,
        'keypoints_3d': sam3d_output.get('pred_keypoints_3d', np.zeros((1, 3))).tolist(),
        'keypoints_2d': sam3d_output.get('pred_keypoints_2d', np.zeros((1, 2))).tolist(),
        'global_rot': sam3d_output.get('global_rot', np.zeros(3)).tolist(),
        'body_pose': sam3d_output.get('body_pose_params', np.zeros(1)).tolist(),
    }
    
    return result


def process_video(video_path, estimator, output_dir, skip_frames=1):
    """
    Обрабатывает видео и извлекает 3D позу для каждого кадра
    
    Args:
        video_path: путь к видео
        estimator: SAM3DBodyEstimator
        output_dir: папка для результатов
        skip_frames: обрабатывать каждый N-й кадр (для ускорения)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
    print(f"⏭️ Processing every {skip_frames} frame(s)")
    
    all_poses = []
    frame_indices = []
    vertices_list = []
    
    frame_count = 0
    processed_count = 0
    
    # Bbox для всего кадра (тренер занимает весь кадр)
    full_frame_bbox = np.array([[0, 0, width, height]], dtype=np.float32)
    
    with tqdm(total=total_frames // skip_frames, desc="Extracting 3D Pose") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                # Конвертируем BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Запускаем inference
                    outputs = estimator.process_one_image(
                        frame_rgb,
                        bboxes=full_frame_bbox,
                        inference_type="body"  # Только тело, без рук (быстрее)
                    )
                    
                    if outputs and len(outputs) > 0:
                        # Берём первого (единственного) человека
                        pose_data = sam3d_to_mediapipe_format(outputs[0])
                        
                        if pose_data:
                            all_poses.append(pose_data)
                            frame_indices.append(frame_count)
                            
                            # Сохраняем vertices для 3D визуализации
                            if 'pred_vertices' in outputs[0]:
                                vertices_list.append(outputs[0]['pred_vertices'])
                    
                except Exception as e:
                    print(f"⚠️ Frame {frame_count}: {e}")
                
                processed_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    print(f"\n✅ Processed {processed_count} frames")
    
    # Сохраняем результаты
    
    # 1. JSON с позами для браузера (сравнение с MediaPipe)
    poses_json = {
        'fps': fps / skip_frames,  # Эффективный FPS
        'original_fps': fps,
        'skip_frames': skip_frames,
        'width': width,
        'height': height,
        'total_frames': len(all_poses),
        'frame_indices': frame_indices,
        'poses': all_poses
    }
    
    json_path = os.path.join(output_dir, 'trainer_poses.json')
    with open(json_path, 'w') as f:
        json.dump(poses_json, f)
    print(f"💾 Saved: {json_path}")
    
    # 2. NPZ с полными данными для 3D визуализации
    npz_path = os.path.join(output_dir, 'trainer_poses.npz')
    
    save_dict = {
        'fps': fps / skip_frames,
        'original_fps': fps,
        'skip_frames': skip_frames,
        'width': width,
        'height': height,
        'frame_indices': np.array(frame_indices),
        'faces': estimator.faces,  # Меш-топология для рендеринга
    }
    
    # Добавляем poses как отдельные массивы
    if all_poses:
        # Извлекаем joints_3d в numpy массив
        joints_3d_list = [p.get('joints_3d', []) for p in all_poses]
        if joints_3d_list and joints_3d_list[0]:
            save_dict['joints_3d'] = np.array(joints_3d_list)
        
        keypoints_3d_list = [p.get('keypoints_3d', []) for p in all_poses]
        if keypoints_3d_list and keypoints_3d_list[0]:
            save_dict['keypoints_3d'] = np.array(keypoints_3d_list)
    
    if vertices_list:
        save_dict['vertices'] = np.array(vertices_list)
    
    np.savez_compressed(npz_path, **save_dict)
    print(f"💾 Saved: {npz_path}")
    
    return json_path, npz_path


def main():
    parser = argparse.ArgumentParser(description='Extract 3D Pose from Video using SAM 3D Body')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output/poses', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam-3d-body-dinov3/model.ckpt',
                        help='Path to SAM 3D Body checkpoint')
    parser.add_argument('--mhr', type=str, default='checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt',
                        help='Path to MHR model')
    parser.add_argument('--skip', type=int, default=1, help='Process every N frames')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Проверяем файлы
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Определяем устройство
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    
    # Инициализируем модель
    estimator = setup_estimator(args.checkpoint, args.mhr, device)
    
    # Обрабатываем видео
    json_path, npz_path = process_video(
        args.video, 
        estimator, 
        args.output,
        skip_frames=args.skip
    )
    
    print(f"\n🎉 Done!")
    print(f"📁 JSON (for browser): {json_path}")
    print(f"📁 NPZ (for 3D viz): {npz_path}")


if __name__ == '__main__':
    main()
