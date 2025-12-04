#!/bin/bash
# Setup script for Python 3.11 environment with SAM 3D Body and dependencies

set -e  # Exit on error

echo "=========================================="
echo "SAM 3D Body Python 3.11 Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Step 1: Creating conda environment 'sam_3d_body' with Python 3.11..."
conda create -n sam_3d_body python=3.11 -y

echo ""
echo "Step 2: Activating environment..."
echo "Note: You'll need to run 'conda activate sam_3d_body' after this script"
echo ""

# The rest of the installation needs to happen in the activated environment
# So we'll create a separate script that the user runs after activation

cat > install_dependencies.sh << 'EOF'
#!/bin/bash
# This script should be run AFTER activating the conda environment
# conda activate sam_3d_body

set -e

echo "Installing dependencies in sam_3d_body environment..."
echo ""

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "sam_3d_body" ]]; then
    echo "Error: Please activate the sam_3d_body environment first:"
    echo "  conda activate sam_3d_body"
    exit 1
fi

echo "Step 1: Installing PyTorch..."
pip install torch torchvision torchaudio

echo ""
echo "Step 2: Installing SAM 3D Body core dependencies..."
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope \
    ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black \
    pycocotools tensorboard huggingface_hub

echo ""
echo "Step 3: Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo ""
echo "Step 4: Installing MediaPipe and comparison dependencies..."
pip install mediapipe scipy fastdtw

echo ""
echo "Step 5: Installing 3D visualization dependencies..."
pip install trimesh pillow opencv-contrib-python

echo ""
echo "Step 6: Installing optional dependencies for background removal..."
pip install rembg

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from sam_3d_body import load_sam_3d_body_hf; print('SAM 3D Body: OK')" 2>/dev/null || echo "SAM 3D Body: Will be available from sam-3d-body directory"

echo ""
echo "All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Test trainer video processing:"
echo "   cd /Users/user/Documents/pers2"
echo "   python process_trainer_video.py --input squat.mp4 --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt --mhr ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt --output squat_poses.npz --max-frames 30"
echo ""
echo "2. Test user pose capture:"
echo "   python capture_user_pose.py --duration 5 --output test_user.npz"
echo ""
echo "3. Run real-time comparison (once implemented):"
echo "   python realtime_comparison.py --coach-poses squat_poses.npz --coach-video squat.mp4"
EOF

chmod +x install_dependencies.sh

echo ""
echo "=========================================="
echo "Environment created successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   conda activate sam_3d_body"
echo ""
echo "2. Run the dependency installation script:"
echo "   ./install_dependencies.sh"
echo ""
echo "The installation script has been created in the current directory."
echo ""
