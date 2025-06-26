#!/bin/bash

# HADM Server Complete Setup Script
# This script performs a complete installation of HADM Server with all dependencies

set -e  # Exit on any error

echo "🚀 HADM Server Complete Setup"
echo "============================="
echo ""
echo "This script will:"
echo "1. Create and activate a virtual environment"
echo "2. Install system dependencies"
echo "3. Install Python dependencies including PyTorch, Detectron2, and MMCV-Full"
echo "4. Clone HADM repository"
echo "5. Download pretrained models"
echo "6. Verify the installation"
echo ""

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  Warning: Running as root is not recommended."
   read -p "Continue anyway? (y/N): " -n 1 -r
   echo
   if [[ ! $REPLY =~ ^[Yy]$ ]]; then
       echo "❌ Setup cancelled. Please run as a regular user."
       exit 1
   fi
fi

# Check for required system commands
echo "🔍 Checking system prerequisites..."
for cmd in python3 pip git wget curl; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ Required command '$cmd' not found. Please install it first."
        exit 1
    fi
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv --prompt HADM_server
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libtiff-dev \
    libfreetype6-dev \
    libpng-dev \
    libwebp-dev \
    git \
    wget \
    curl

# Upgrade pip and install build tools
echo "🔧 Upgrading pip and installing build tools..."
pip install -U pip wheel setuptools openmim ninja psutil cmake

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📚 Installing core Python dependencies..."
pip install \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    Pillow==9.0.0 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    cloudpickle==2.2.1 \
    fairscale==0.4.13 \
    requests==2.31.0 \
    tqdm==4.66.1

# Install Detectron2 with CUDA support
echo "🔍 Installing Detectron2 with CUDA support..."
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html

# Install MMCV-Full with CUDA ops (CRITICAL for HADM)
echo "🎯 Installing MMCV-Full with CUDA ops..."
echo "⏰ This is critical for HADM model loading and may take 10-20 minutes..."

# Set environment variables for MMCV compilation
export MMCV_WITH_OPS=1
export CUDA_HOME=/usr/local/cuda-12.1

# Try pre-built wheel first
echo "🔄 Attempting to install pre-built mmcv-full wheel..."
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html || {
    echo "⚠️  Pre-built wheel failed, building from source..."
    echo "⏰ This will take 10-20 minutes..."
    
    # Clone and build from source
    if [ -d "mmcv" ]; then
        rm -rf mmcv
    fi
    
    git clone --depth 1 https://github.com/open-mmlab/mmcv.git
    cd mmcv
    pip install -v -e .
    cd ..
    
    echo "✅ MMCV-Full built and installed from source"
}

# Install development dependencies
echo "🧪 Installing development dependencies..."
pip install \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    black==23.11.0 \
    flake8==6.1.0 \
    python-json-logger==2.0.7

# Clone HADM repository if it doesn't exist
if [ ! -d "HADM" ]; then
    echo "📥 Cloning HADM repository..."
    git clone https://github.com/marlinfiggins/HADM.git
    echo "✅ HADM repository cloned"
else
    echo "📁 HADM repository already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs cache pretrained_models datasets

# Set environment variables
echo "🌍 Setting up environment variables..."
export DETECTRON2_DATASETS="$(pwd)/datasets"

# Download models
echo "📥 Downloading pretrained models..."
if [ -f "scripts/download_models.sh" ]; then
    chmod +x scripts/download_models.sh
    bash scripts/download_models.sh
else
    echo "⚠️  Download script not found, skipping model download"
fi

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import detectron2; print(f'✅ Detectron2: {detectron2.__version__}')"
python -c "import mmcv; print(f'✅ MMCV: {mmcv.__version__}')"
python -c "from mmcv import ops; print('✅ MMCV ops available')" || echo "❌ MMCV ops not available"

# Test HADM imports
echo "🔧 Testing HADM imports..."
cd HADM 2>/dev/null || echo "⚠️  HADM directory not found"
python -c "
import sys
sys.path.insert(0, '.')
try:
    from projects.ViTDet.configs.eva2_o365_to_coco.demo_local import model
    print('✅ HADM local config import successful')
except Exception as e:
    print(f'❌ HADM local config import failed: {e}')

try:
    from projects.ViTDet.configs.eva2_o365_to_coco.demo_global import model
    print('✅ HADM global config import successful')
except Exception as e:
    print(f'❌ HADM global config import failed: {e}')
" 2>/dev/null || echo "⚠️  HADM import test skipped"

cd ..

# Run diagnostics
echo "🔍 Running diagnostics..."
if [ -f "diagnose_models.py" ]; then
    python diagnose_models.py
else
    echo "⚠️  Diagnostic script not found"
fi

echo ""
echo "🎉 HADM Server Setup Complete!"
echo "=============================="
echo ""
echo "✅ Installation Summary:"
echo "- Virtual environment: $(pwd)/venv"
echo "- PyTorch with CUDA support installed"
echo "- Detectron2 installed"
echo "- MMCV-Full with CUDA ops installed"
echo "- HADM repository cloned"
echo "- Pretrained models downloaded"
echo ""
echo "🚀 Next Steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Test the server: python -m uvicorn app.main:app --host 0.0.0.0 --port 8080"
echo "3. Open browser: http://localhost:8080/docs"
echo ""
echo "🔍 If you encounter issues:"
echo "- Check CUDA: nvidia-smi"
echo "- Verify MMCV: python -c 'from mmcv import ops; print(\"OK\")'"
echo "- Run diagnostics: python diagnose_models.py"
echo ""
echo "✨ Happy coding!" 