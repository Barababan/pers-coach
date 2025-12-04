"""
Alternative trainer video processing using SAM 3D Body notebook utilities.
This version uses the simplified setup from the notebook examples.
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add sam-3d-body to path
sam_3d_path = Path(__file__).parent / "sam-3d-body"
if sam_3d_path.exists():
    sys.path.insert(0, str(sam_3d_path))

try:
    from notebook.utils import setup_sam_3d_body
    from tools.vis_utils import visualize_sample_together
except ImportError as e:
    print(f"Error importing SAM 3D Body: {e}")
    print("\nThis script requires Python 3.10+ due to SAM 3D Body dependencies.")
    print("Please create a conda environment with Python 3.11:")
    print("  conda create -n sam_3d_body python=3.11 -y")
    print("  conda activate sam_3d_body")
    print("  # Then install dependencies as per INSTALL.md")
    sys.exit(1)

from pose_utils import save_pose_sequence, calculate_key_joint_angles


def process_video_simple(
    video_path: str,
    output_path: str,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    frame_skip: int = 1,
    max_frames: int = None
):
    """
    Process video using simplified notebook setup.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output (.npz file)
        hf_repo_id: HuggingFace repo ID for model
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Determine device
    import torch
    # Force CPU for stability as MPS is causing memory issues/freezing
    device = "cpu" 
    print(f"Using device: {device}")

    # Setup estimator using notebook utility
    # Disable MoGe FOV estimator for speed on CPU
    estimator = setup_sam_3d_body(
        hf_repo_id=hf_repo_id,
        device=device,
        fov_name=None
    )
    
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
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        if max_frames and processed_count >= max_frames:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Process frame
            outputs = estimator.process_one_image(frame_rgb)
            
            if len(outputs) == 0:
                pose_sequence.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detected': False
                })
            else:
                output = outputs[0]
                
                pose_data = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'detected': True,
                    'keypoints_3d': output['pred_keypoints_3d'],
                    'keypoints_2d': output['pred_keypoints_2d'],
                    'vertices': output['pred_vertices'],
                    'focal_length': output['focal_length'],
                    'cam_t': output['pred_cam_t'],
                    'global_rot': output['global_rot'],
                    'joint_coords': output['pred_joint_coords'],
                    'global_rots': output['pred_global_rots'],
                    'bbox': output['bbox']
                }
                
                try:
                    angles = calculate_key_joint_angles(output['pred_keypoints_3d'])
                    pose_data['joint_angles'] = angles
                except Exception as e:
                    print(f"\nWarning: Could not calculate angles for frame {frame_idx}: {e}")
                
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
    
    # Save
    metadata = {
        'video_path': video_path,
        'fps': fps,
        'resolution': (width, height),
        'total_frames': total_frames,
        'frame_skip': frame_skip,
        'frames_processed': len(pose_sequence)
    }
    
    save_pose_sequence(output_path, pose_sequence, metadata)
    
    detected_frames = sum(1 for p in pose_sequence if p.get('detected', False))
    print(f"\nProcessing complete!")
    print(f"  Frames with detected pose: {detected_frames}/{len(pose_sequence)}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process trainer video (simplified version)")
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output .npz file')
    parser.add_argument('--repo', default='facebook/sam-3d-body-dinov3', help='HuggingFace repo ID')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_poses.npz")
    
    process_video_simple(
        video_path=args.input,
        output_path=args.output,
        hf_repo_id=args.repo,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
