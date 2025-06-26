# Install Missing Dependencies

Based on your test results, the models are failing to load because of missing dependencies. Here's how to fix it:

## 🔧 System Dependencies (if not already installed)

```bash
# Ubuntu/Debian - Image processing system libraries
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev libpng-dev libopencv-dev

# CentOS/RHEL/Fedora
sudo yum install -y libjpeg-turbo-devel zlib-devel libtiff-devel freetype-devel libpng-devel opencv-devel

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel
```

## 🔧 Critical Python Dependencies to Install

### 1. Install Detectron2 (CRITICAL)

```bash
# For CUDA 12.8 (matching your PyTorch installation)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu128/torch2.0/index.html
```

### 2. Install MMCV (CRITICAL)

```bash
# Install mmcv for computer vision operations
pip install mmcv==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.0/index.html
```

### 3. Install Pillow 9.0.0 (CRITICAL - Compatibility Fix)

```bash
# HADM requires older Pillow version for Image.LINEAR compatibility
pip install Pillow==9.0.0
```

### 4. Install Additional CV Dependencies

```bash
# Install remaining computer vision dependencies
pip install opencv-python==4.8.1.78
pip install timm==0.5.4
pip install fvcore
pip install omegaconf
pip install cloudpickle  # Required by detectron2
```

## 🧪 Test Installation

After installing dependencies, run the diagnostic script:

```bash
python diagnose_models.py
```

This will check:
- ✅ All dependencies are installed
- ✅ HADM configurations can be imported  
- ✅ Model files can be loaded
- ✅ CUDA is working (if available)

## 🚀 Test API After Installation

Once dependencies are installed, restart your server and test:

```bash
# Test the API endpoints
python test_api.py
```

You should see:
- ✅ Health check: Models loaded = True
- ✅ Detection endpoints working
- ✅ Actual detection results (even if empty for cat image)

## 🔍 Troubleshooting

### If Detectron2 Installation Fails

Try the CPU version first:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

### If MMCV Installation Fails

Try without specifying CUDA version:
```bash
pip install mmcv==1.7.1
```

### Check Installation

```bash
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
python -c "import mmcv; print('MMCV:', mmcv.__version__)"
```

## 📋 Installation Order

1. **First**: Install PyTorch (already done ✅)
2. **Second**: Install Pillow 9.0.0 (HADM compatibility fix)
3. **Third**: Install Detectron2 
4. **Fourth**: Install MMCV
5. **Fifth**: Install other CV dependencies
6. **Finally**: Test with diagnostic script

## 🎯 Expected Results

After successful installation:
- Health endpoint shows `models_loaded: {"local": true, "global": true}`
- Detection endpoints return proper responses (not 500 errors)
- Models can process images (even if no artifacts detected in cat photos) 