# Real-Time Pose Comparison System - Quick Start Guide

## Overview

A real-time 3-panel comparison system that shows:
1. **Left Panel**: 3D pose visualization (coach in blue + user in green)
2. **Center Panel**: Coach video playback
3. **Right Panel**: Live user webcam feed

All on a common gray background with real-time similarity scoring.

## Prerequisites

### Python 3.11 Required

SAM 3D Body requires Python 3.10+. Install Python 3.11 using one of these methods:

**Option 1: Using pyenv (Recommended)**
```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# Verify
python --version  # Should show 3.11.x
```

**Option 2: Using Homebrew**
```bash
brew install python@3.11
```

**Option 3: Download from python.org**
Visit https://www.python.org/downloads/ and download Python 3.11

### Install Dependencies

```bash
cd /Users/user/Documents/pers2

# Core dependencies
pip install torch torchvision torchaudio
pip install mediapipe opencv-python scipy fastdtw
pip install matplotlib pillow

# SAM 3D Body dependencies (for trainer processing)
pip install pytorch-lightning pyrender yacs scikit-image einops timm \
    dill pandas rich hydra-core networkx==3.2.1 roma joblib \
    huggingface_hub tensorboard
```

## Quick Start

### Step 1: Process Trainer Video

First, extract 3D poses from the trainer video:

```bash
cd /Users/user/Documents/pers2

python process_trainer_video_simple.py \
  --input squat.mp4 \
  --output squat_poses.npz \
  --max-frames 100
```

This creates `squat_poses.npz` with the trainer's 3D pose sequence.

### Step 2: Run Real-Time Comparison

```bash
python realtime_comparison.py \
  --coach-poses squat_poses.npz \
  --coach-video squat.mp4
```

This opens a 3-panel window:
- **Left**: 3D skeletons (coach=blue, user=green)
- **Center**: Coach video
- **Right**: Your webcam

### Controls

- **SPACE**: Play/Pause coach video
- **R**: Restart from beginning
- **Q**: Quit
- **+/-**: Adjust playback speed

## Features

### Real-Time Feedback

The system provides:
- **Similarity Score** (0-100): Overall match quality
- **Joint Angle Feedback**: Shows top 3 joints that need correction
- **Color-Coded Feedback**:
  - 🟢 Green: Good match (< 15° difference)
  - 🟡 Yellow: Fair match (15-30° difference)
  - 🔴 Red: Needs improvement (> 30° difference)

### 3D Visualization Options

**Simple Renderer** (Default - Faster)
```bash
python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4
```

**Matplotlib 3D Renderer** (Better Quality)
```bash
python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4 --use-3d
```

## File Structure

```
/Users/user/Documents/pers2/
├── pose_utils.py                    # Utility functions
├── pose_renderer_3d.py              # 3D skeleton rendering
├── realtime_comparison.py           # Main application
├── process_trainer_video.py         # Trainer processing (full version)
├── process_trainer_video_simple.py  # Trainer processing (simplified)
├── capture_user_pose.py             # Standalone user capture
├── compare_poses.py                 # Offline comparison
├── README_POSE_COMPARISON.md        # Detailed documentation
└── checkpoints/
    └── sam-3d-body-dinov3/         # SAM 3D Body model (2.1 GB)
```

## Troubleshooting

### "No module named 'sam_3d_body'"

Make sure you're in the correct directory and the `sam-3d-body` folder exists:
```bash
cd /Users/user/Documents/pers2
ls sam-3d-body  # Should show the repository contents
```

### "Could not open camera"

Try a different camera ID:
```bash
python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4 --camera 1
```

### Low Frame Rate

- Use the simple renderer (default) instead of `--use-3d`
- Close other applications
- Reduce video resolution

### Python Version Error

Verify you're using Python 3.11:
```bash
python --version
```

If not, activate the correct Python environment or use `python3.11` explicitly.

## Advanced Usage

### Process Only Part of Video

```bash
python process_trainer_video_simple.py \
  --input squat.mp4 \
  --output squat_poses.npz \
  --max-frames 50 \
  --frame-skip 2  # Process every 2nd frame
```

### Offline Comparison

For analyzing recorded sessions:
```bash
# 1. Capture user session
python capture_user_pose.py --duration 10 --output my_squat.npz

# 2. Compare offline
python compare_poses.py \
  --trainer squat_poses.npz \
  --user my_squat.npz \
  --output results.npz
```

## Next Steps

1. **Add Background Removal**: Integrate `rembg` for user video background removal
2. **Save Sessions**: Record comparison sessions for review
3. **Multi-Angle**: Support multiple camera views
4. **Audio Feedback**: Add voice cues for corrections
5. **Progress Tracking**: Track improvement over time

## Support

For detailed documentation, see:
- [README_POSE_COMPARISON.md](file:///Users/user/Documents/pers2/README_POSE_COMPARISON.md) - Full documentation
- [implementation_plan.md](file:///Users/user/.gemini/antigravity/brain/145c71e0-fe89-49a7-9f50-eaa00c092b09/implementation_plan.md) - Technical architecture
