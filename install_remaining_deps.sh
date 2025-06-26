#!/bin/bash
# Install remaining dependencies for HADM

echo "🔧 Installing Remaining HADM Dependencies"
echo "=========================================="

# Install fairscale (required by detectron2)
echo "⚖️ Installing fairscale..."
pip install fairscale

# Install additional dependencies that might be needed
echo "📦 Installing additional dependencies..."
pip install fvcore omegaconf hydra-core iopath
pip install pycocotools  # If not already installed

# Try different detectron2 installation methods
echo "🔍 Installing detectron2..."

# Method 1: Try pre-built wheel for different CUDA versions
echo "  Trying CUDA 12.1 wheel..."
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html

# If that fails, try CPU version
if [ $? -ne 0 ]; then
    echo "  CUDA wheel failed, trying CPU version..."
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
fi

# If that fails, try building from source
if [ $? -ne 0 ]; then
    echo "  Pre-built wheels failed, trying to build from source..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
fi

# If that fails, use the HADM detectron2
if [ $? -ne 0 ]; then
    echo "  GitHub install failed, using HADM detectron2..."
    cd HADM
    pip install -e .
    cd ..
fi

# Install MMCV
echo "📊 Installing MMCV..."
pip install mmcv==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html

# If MMCV fails, try without CUDA specification
if [ $? -ne 0 ]; then
    echo "  CUDA MMCV failed, trying generic version..."
    pip install mmcv==1.7.1
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Test with:"
echo "python diagnose_models.py" 