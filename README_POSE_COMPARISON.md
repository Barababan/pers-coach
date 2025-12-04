# SAM 3D Body & MediaPipe Pose Comparison System

A system for comparing user movements against trainer reference videos using SAM 3D Body and MediaPipe 3D Pose Estimation.

## Overview

This system consists of three main components:

1. **Trainer Video Processing** (`process_trainer_video.py`): Uses SAM 3D Body to extract 3D pose data from trainer videos
2. **User Pose Capture** (`capture_user_pose.py`): Uses MediaPipe to capture user movements in real-time with world landmarks
3. **Pose Comparison** (`compare_poses.py`): Compares user poses against trainer reference with similarity scoring

## Prerequisites

### Required Dependencies

```bash
# Core dependencies (already installed)
python3 -m pip install mediapipe torch torchvision pytorch-lightning

# Additional dependencies for comparison
python3 -m pip install scipy fastdtw
```

### SAM 3D Body Setup

1. **Request HuggingFace Access**: Visit [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and request access

2. **Download Model Checkpoints**:
```bash
# Install HuggingFace CLI
python3 -m pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Download checkpoints
cd /Users/user/Documents/pers2
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

## Usage

### 1. Process Trainer Video

Extract 3D pose reference from trainer video:

```bash
python3 process_trainer_video.py \
  --input squat.mp4 \
  --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --output trainer_reference.npz
```

**Options**:
- `--frame-skip N`: Process every Nth frame (default: 1)
- `--max-frames N`: Limit processing to N frames
- `--bbox-thresh 0.8`: Bounding box detection threshold

### 2. Capture User Pose

Record user movements with webcam:

```bash
# Capture for 10 seconds
python3 capture_user_pose.py --duration 10 --output user_pose.npz

# Or capture until 'q' is pressed
python3 capture_user_pose.py --output user_pose.npz
```

**Options**:
- `--duration N`: Record for N seconds
- `--camera 0`: Camera device ID
- `--no-preview`: Disable live preview window

### 3. Compare Poses

Compare user movements against trainer reference:

```bash
python3 compare_poses.py \
  --trainer trainer_reference.npz \
  --user user_pose.npz \
  --output comparison_results.npz
```

**Options**:
- `--no-dtw`: Disable Dynamic Time Warping alignment (use frame-by-frame)

## Output

The comparison script provides:

- **Overall Similarity Score** (0-100): Higher is better
- **MPJPE** (Mean Per Joint Position Error): Lower is better
- **Per-Joint Angle Analysis**: Shows which joints match well and which need improvement
- **Match Quality**: GOOD (<15°), FAIR (15-30°), POOR (>30°)

Example output:
```
============================================================
COMPARISON RESULTS
============================================================
Total frame comparisons: 145

Overall Similarity Score: 78.5/100
  (std: 8.2)

Mean Position Error (MPJPE): 0.0234
Mean Angle Difference: 18.3°
============================================================

Per-Joint Angle Analysis:
------------------------------------------------------------
  left_elbow          :   12.3° ±   5.2° [GOOD]
  right_elbow         :   14.8° ±   6.1° [GOOD]
  left_knee           :   22.5° ±   8.3° [FAIR]
  right_knee          :   19.7° ±   7.5° [FAIR]
  left_hip            :   16.2° ±   5.9° [FAIR]
  right_hip           :   15.8° ±   6.2° [FAIR]
============================================================
```

## File Structure

```
/Users/user/Documents/pers2/
├── pose_utils.py              # Shared utility functions
├── process_trainer_video.py   # Trainer video processing
├── capture_user_pose.py       # User pose capture
├── compare_poses.py           # Pose comparison
├── sam-3d-body/              # SAM 3D Body repository
└── checkpoints/              # Model checkpoints (download separately)
    └── sam-3d-body-dinov3/
        ├── model.ckpt
        └── assets/
            └── mhr_model.pt
```

## Technical Details

### Coordinate Systems

- **SAM 3D Body**: Uses MHR (Momentum Human Rig) with 3D keypoints in camera space
- **MediaPipe**: Uses world landmarks in meters, relative to hip center

Both systems provide 33 body landmarks. The comparison normalizes poses by:
1. Centering at pelvis (hip center)
2. Scaling to unit torso length
3. Comparing joint angles and positions

### Temporal Alignment

The system uses **Dynamic Time Warping (DTW)** to align sequences of different speeds. This allows comparing movements even if the user performs them faster or slower than the trainer.

### Similarity Metrics

1. **Position Score**: Based on Mean Per Joint Position Error (MPJPE)
2. **Angle Score**: Based on joint angle differences
3. **Overall Score**: Average of position and angle scores

## Troubleshooting

### "No human detected in frame"
- Ensure good lighting
- Make sure full body is visible in frame
- Adjust `--bbox-thresh` (try lower values like 0.5)

### "Could not align sequences"
- Ensure both sequences have detected poses
- Check that movements are similar enough
- Try `--no-dtw` for simpler comparison

### Import errors
- Verify all dependencies are installed
- Check that `sam-3d-body` directory exists
- Ensure Python path includes sam-3d-body

## Next Steps

- Add real-time comparison mode
- Create visualization of pose differences
- Add audio/visual feedback during capture
- Support multiple camera angles
- Export comparison results to video
