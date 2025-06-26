#!/bin/bash

# HADM Server Environment Setup Script
set -e

echo "🚀 Setting up HADM Server environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv --prompt HADM_server
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install xformers
echo "🚀 Installing xformers..."
pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.18#egg=xformers

# Install mmcv and related packages
echo "📊 Installing mmcv..."
pip install mmcv==1.7.1 openmim
mim install mmcv-full

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Install detectron2 from HADM directory
echo "🔍 Installing detectron2 from HADM..."
cd HADM
python -m pip install -e .
cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p pretrained_models
mkdir -p datasets
mkdir -p logs
mkdir -p cache

# Set environment variables
echo "🌍 Setting up environment variables..."
export DETECTRON2_DATASETS=./datasets

# Copy environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file. Please configure it according to your setup."
fi

echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Download pretrained models (see scripts/download_models.sh)"
echo "2. Configure your .env file"
echo "3. Run the server: uvicorn app.main:app --host 0.0.0.0 --port 8080" 