"""
Process trainer videos using SAM 3D Body to extract 3D pose estimation.
Creates a reference "etalon" (standard) for movement comparison.
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Add sam-3d-body to path
sam_3d_path = Path(__file__).parent / "sam-3d-body"
if sam_3d_path.exists():
    sys.path.insert(0, str(sam_3d_path))

try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from tools.build_detector import HumanDetector
except ImportError as e:
    print(f"Error importing SAM 3D Body: {e}")
    print("Make sure sam-3d-body is properly installed and checkpoints are downloaded.")
    sys.exit(1)

from pose_utils import save_pose_sequence, calculate_key_joint_angles


def setup_sam_3d_estimator(checkpoint_path: str, mhr_path: str, device: str = 'cuda'):
    """
    Initialize SAM 3D Body estimator.
    
    Args:
        checkpoint_path: Path to SAM 3D Body checkpoint
        mhr_path: Path to MHR model
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        SAM3DBodyEstimator instance
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAM 3D Body model
    print("Loading SAM 3D Body model...")
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    
    # Initialize human detector
    print("Loading human detector...")
    detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
    human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)
    
    # Create estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,  # Not needed for video processing
        fov_estimator=None     # Will use default FOV
    )
    
    print("SAM 3D Body estimator ready!")
    return estimator


def process_video_to_poses(
    video_path: str,
    estimator: SAM3DBodyEstimator,
    output_path: str,
    frame_skip: int = 1,
    max_frames: int = None,
    bbox_thresh: float = 0.8
):
    """
    Process a video and extract 3D poses for each frame.
    
    Args:
        video_path: Path to input video
        estimator: SAM3DBodyEstimator instance
        output_path: Path to save output (.npz file)
        frame_skip: Process every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to process (None = all)
        bbox_thresh: Bounding box detection threshold
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    frames_to_process = total_frames // frame_skip
    print(f"  Processing every {frame_skip} frame(s): {frames_to_process} frames total\n")
    
    # Storage for pose data
    pose_sequence = []
    frame_idx = 0
    processed_count = 0
    
    # Process video
    pbar = tqdm(total=frames_to_process, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        if max_frames and processed_count >= max_frames:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run SAM 3D Body estimation
        try:
            outputs = estimator.process_one_image(
                frame_rgb,
                bbox_thr=bbox_thresh,
                use_mask=False
            )
            
            if len(outputs) == 0:
                print(f"\nWarning: No human detected in frame {frame_idx}")
                # Store empty data for this frame
                pose_sequence.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detected': False
                })
            else:
                # Use first detected person
                output = outputs[0]
                
                # Extract key pose data
                pose_data = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detected': True,
                    'keypoints_3d': output['pred_keypoints_3d'],  # 3D keypoints
                    'keypoints_2d': output['pred_keypoints_2d'],  # 2D projections
                    'vertices': output['pred_vertices'],          # Mesh vertices
                    'focal_length': output['focal_length'],
                    'cam_t': output['pred_cam_t'],               # Camera translation
                    'global_rot': output['global_rot'],          # Global rotation
                    'joint_coords': output['pred_joint_coords'], # Joint coordinates
                    'global_rots': output['pred_global_rots'],   # Joint rotations
                    'bbox': output['bbox']
                }
                
                # Calculate joint angles for this frame
                try:
                    angles = calculate_key_joint_angles(output['pred_keypoints_3d'])
                    pose_data['joint_angles'] = angles
                except Exception as e:
                    print(f"\nWarning: Could not calculate joint angles for frame {frame_idx}: {e}")
                
                pose_sequence.append(pose_data)
                
        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {e}")
            pose_sequence.append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'detected': False,
                'error': str(e)
            })
        
        frame_idx += 1
        processed_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save pose sequence
    metadata = {
        'video_path': video_path,
        'fps': fps,
        'resolution': (width, height),
        'total_frames': total_frames,
        'frame_skip': frame_skip,
        'frames_processed': len(pose_sequence),
        'bbox_threshold': bbox_thresh
    }
    
    save_pose_sequence(output_path, pose_sequence, metadata)
    
    # Print summary
    detected_frames = sum(1 for p in pose_sequence if p.get('detected', False))
    print(f"\nProcessing complete!")
    print(f"  Frames with detected pose: {detected_frames}/{len(pose_sequence)}")
    print(f"  Output saved to: {output_path}")
    
    return pose_sequence, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Process trainer video with SAM 3D Body to create pose reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings
  python process_trainer_video.py --input squat.mp4 --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
  
  # Process every 2nd frame
  python process_trainer_video.py --input squat.mp4 --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt --frame-skip 2
  
  # Process first 100 frames only
  python process_trainer_video.py --input squat.mp4 --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt --max-frames 100
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output .npz file path (default: <input>_poses.npz)')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM 3D Body checkpoint')
    parser.add_argument('--mhr', required=True, help='Path to MHR model file')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame (default: 1)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (default: all)')
    parser.add_argument('--bbox-thresh', type=float, default=0.8, help='Bounding box threshold (default: 0.8)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_poses.npz")
    
    # Setup estimator
    estimator = setup_sam_3d_estimator(args.checkpoint, args.mhr, args.device)
    
    # Process video
    process_video_to_poses(
        video_path=args.input,
        estimator=estimator,
        output_path=args.output,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        bbox_thresh=args.bbox_thresh
    )


if __name__ == '__main__':
    main()
