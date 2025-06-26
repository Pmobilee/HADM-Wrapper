# PyTorch Installation Guide

## CUDA 12.8 Installation

All PyTorch installations in this project use CUDA 12.8. Use the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Updated Files

The following files have been updated to use CUDA 12.8:

- `setup.sh` - Main setup script
- `scripts/setup_environment.sh` - Environment setup script  
- `Dockerfile` - Docker container build
- `requirements.txt` - Installation instructions in comments
- `SETUP_SUMMARY.md` - Documentation

## Verification

After installation, verify PyTorch is using CUDA 12.8:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Alternative Installation

If you need a different CUDA version, visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to generate the appropriate installation command.

## Compatibility

- **CUDA 12.8** is compatible with most modern GPUs
- Requires NVIDIA driver version 450.80.02 or newer
- Supports compute capability 3.5 and above 