#!/bin/bash

# HADM Server Complete Setup Script
# This script performs all setup steps in one go

set -e  # Exit on any error

echo "🚀 HADM Server Complete Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_warning "This script should not be run as root for security reasons"
   read -p "Continue anyway? (y/N): " -n 1 -r
   echo
   if [[ ! $REPLY =~ ^[Yy]$ ]]; then
       exit 1
   fi
fi

# Step 1: System Dependencies
print_status "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip git wget curl build-essential cmake
    print_success "System dependencies installed"
elif command -v yum &> /dev/null; then
    sudo yum update -y
    sudo yum install -y python3 python3-venv python3-pip git wget curl gcc gcc-c++ cmake
    print_success "System dependencies installed"
else
    print_warning "Could not detect package manager. Please install: python3, python3-venv, python3-pip, git, wget, curl, build-essential, cmake"
fi

# Step 2: Create and activate virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv --prompt HADM_server
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Step 3: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
print_success "Pip upgraded"

# Step 4: Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
print_success "PyTorch installed"

# Step 5: Install other requirements
print_status "Installing Python requirements..."
pip install -r requirements.txt
print_success "Requirements installed"

# Step 6: Install xformers (if needed)
print_status "Installing xformers..."
pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.18#egg=xformers || print_warning "xformers installation failed, continuing..."

# Step 7: Install mmcv and related packages
print_status "Installing mmcv..."
pip install mmcv==1.7.1 openmim
mim install mmcv-full || print_warning "mmcv-full installation failed, continuing..."

# Step 8: Install detectron2 from HADM directory
print_status "Installing detectron2 from HADM..."
if [ -d "HADM" ]; then
    cd HADM
    python -m pip install -e .
    cd ..
    print_success "Detectron2 installed"
else
    print_error "HADM directory not found. Please ensure the HADM repository is cloned."
    exit 1
fi

# Step 9: Create necessary directories
print_status "Creating necessary directories..."
mkdir -p pretrained_models
mkdir -p datasets
mkdir -p logs
mkdir -p cache
print_success "Directories created"

# Step 10: Set environment variables
print_status "Setting up environment variables..."
export DETECTRON2_DATASETS=./datasets

# Step 11: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp env.example .env
    print_success ".env file created"
else
    print_warning ".env file already exists"
fi

# Step 12: Download models (optional)
print_status "Checking for model files..."
if [ ! -f "pretrained_models/eva02_L_coco_det_sys_o365.pth" ]; then
    print_warning "EVA-02-L model not found"
    read -p "Download EVA-02-L model now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading EVA-02-L model..."
        cd pretrained_models
        wget -O eva02_L_coco_det_sys_o365.pth \
            "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_coco_det_sys_o365.pth"
        cd ..
        print_success "EVA-02-L model downloaded"
    fi
fi

# Check for HADM models
if [ ! -f "pretrained_models/HADM-L_0249999.pth" ] || [ ! -f "pretrained_models/HADM-G_0249999.pth" ]; then
    print_warning "HADM model files not found"
    echo "Please download manually:"
    echo "  - HADM-L: https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth"
    echo "  - HADM-G: https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth"
fi

# Step 13: Test installation
print_status "Testing FastAPI installation..."
python -c "from app.main import app; print('✅ FastAPI app imports successfully')" || print_error "FastAPI app import failed"

# Step 14: Make scripts executable
print_status "Setting script permissions..."
chmod +x scripts/setup_environment.sh
chmod +x scripts/download_models.sh
chmod +x setup.sh
print_success "Script permissions set"

# Summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
print_success "HADM Server setup completed successfully"
echo ""
echo "📋 Next Steps:"
echo "1. Download missing model files if needed"
echo "2. Configure your .env file"
echo "3. Start the server:"
echo "   source venv/bin/activate"
echo "   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "4. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "📚 Documentation:"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/api/v1/health"
echo "   - Info: http://localhost:8000/api/v1/info"
echo ""

# Final check
if [ -f "pretrained_models/eva02_L_coco_det_sys_o365.pth" ] && 
   [ -f "pretrained_models/HADM-L_0249999.pth" ] && 
   [ -f "pretrained_models/HADM-G_0249999.pth" ]; then
    print_success "All model files are present! 🚀"
else
    print_warning "Some model files are missing. Server will start but detection may not work."
fi

print_success "Setup script completed! 🎯" 