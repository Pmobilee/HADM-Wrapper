#!/usr/bin/env python3
"""
HADM Model Diagnostics Script
Helps identify missing dependencies and configuration issues.
"""

import sys
import os
from pathlib import Path

# Add HADM to Python path
HADM_PATH = Path(__file__).parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

def check_python_environment():
    """Check Python environment."""
    print("🐍 Python Environment Check")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"HADM path: {HADM_PATH}")
    print(f"HADM exists: {HADM_PATH.exists()}")
    print()

def check_system_dependencies():
    """Check system-level dependencies."""
    print("🔧 System Dependencies Check")
    print("=" * 50)
    
    # Check for common image processing libraries
    import subprocess
    import shutil
    
    system_deps = [
        ("pkg-config", "pkg-config"),
        ("gcc", "GCC compiler"),
        ("g++", "G++ compiler"),
    ]
    
    for cmd, name in system_deps:
        if shutil.which(cmd):
            print(f"✅ {name} - Available")
        else:
            print(f"❌ {name} - Missing")
    
    # Check for image libraries (Ubuntu/Debian style)
    try:
        result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            packages = result.stdout
            image_libs = [
                ("libjpeg", "JPEG library"),
                ("zlib", "Zlib compression"),
                ("libtiff", "TIFF library"),
                ("libfreetype", "FreeType font library"),
                ("libpng", "PNG library"),
            ]
            
            for lib, name in image_libs:
                if lib in packages:
                    print(f"✅ {name} - Available")
                else:
                    print(f"⚠️ {name} - May be missing")
        else:
            print("ℹ️ Could not check system packages (non-Debian system)")
    except:
        print("ℹ️ Could not check system packages")
    
    print()

def check_basic_dependencies():
    """Check basic dependencies."""
    print("📦 Basic Dependencies Check")
    print("=" * 50)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("cloudpickle", "CloudPickle"),
    ]
    
    for module, name in dependencies:
        try:
            exec(f"import {module}")
            if module == "PIL":
                import PIL
                version = PIL.__version__
                if version.startswith("9.0"):
                    print(f"✅ {name} - Available (v{version}) ✅ HADM Compatible")
                else:
                    print(f"⚠️ {name} - Available (v{version}) ⚠️ May need v9.0.0 for HADM")
            else:
                print(f"✅ {name} - Available")
        except ImportError as e:
            print(f"❌ {name} - Missing: {e}")
    print()

def check_detectron2():
    """Check Detectron2 installation."""
    print("🔍 Detectron2 Check")
    print("=" * 50)
    
    try:
        import detectron2
        print(f"✅ Detectron2 - Available (version: {detectron2.__version__})")
        
        # Check specific modules
        detectron2_modules = [
            ("detectron2.config", "Config"),
            ("detectron2.engine", "Engine"),
            ("detectron2.data", "Data"),
            ("detectron2.utils.visualizer", "Visualizer"),
        ]
        
        for module, name in detectron2_modules:
            try:
                exec(f"import {module}")
                print(f"  ✅ {name} - Available")
            except ImportError as e:
                print(f"  ❌ {name} - Missing: {e}")
                
    except ImportError as e:
        print(f"❌ Detectron2 - Missing: {e}")
        print("  💡 Install with: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu128/torch2.0/index.html")
    print()

def check_hadm_structure():
    """Check HADM directory structure."""
    print("📁 HADM Structure Check")
    print("=" * 50)
    
    expected_paths = [
        "projects",
        "projects/ViTDet",
        "projects/ViTDet/configs",
        "projects/ViTDet/configs/eva2_o365_to_coco",
        "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py",
        "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py",
        "detectron2",
    ]
    
    for path in expected_paths:
        full_path = HADM_PATH / path
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - Missing")
    print()

def check_model_files():
    """Check model files."""
    print("🤖 Model Files Check")
    print("=" * 50)
    
    model_dir = Path("pretrained_models")
    expected_models = [
        "eva02_L_coco_det_sys_o365.pth",
        "HADM-L_0249999.pth", 
        "HADM-G_0249999.pth"
    ]
    
    print(f"Model directory: {model_dir.absolute()}")
    print(f"Model directory exists: {model_dir.exists()}")
    
    if model_dir.exists():
        print("Available files:")
        for file in model_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  📄 {file.name} ({size_mb:.1f} MB)")
        
        print("\nExpected models:")
        for model in expected_models:
            model_path = model_dir / model
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {model} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {model} - Missing")
    print()

def check_hadm_imports():
    """Check HADM-specific imports."""
    print("🔧 HADM Imports Check")
    print("=" * 50)
    
    hadm_imports = [
        ("projects.ViTDet.configs.eva2_o365_to_coco.demo_local", "Local config"),
        ("projects.ViTDet.configs.eva2_o365_to_coco.demo_global", "Global config"),
    ]
    
    for module, name in hadm_imports:
        try:
            exec(f"import {module}")
            print(f"✅ {name} - Available")
        except ImportError as e:
            print(f"❌ {name} - Failed: {e}")
    print()

def check_cuda():
    """Check CUDA availability."""
    print("🚀 CUDA Check")
    print("=" * 50)
    
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("💡 CUDA not available - will use CPU")
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
    print()

def test_simple_model_loading():
    """Test simple model loading."""
    print("🧪 Simple Model Loading Test")
    print("=" * 50)
    
    model_dir = Path("pretrained_models")
    
    for model_name in ["HADM-L_0249999.pth", "HADM-G_0249999.pth"]:
        model_path = model_dir / model_name
        if model_path.exists():
            try:
                import torch
                print(f"Testing {model_name}...")
                
                # Try to load model weights (handle PyTorch 2.6 weights_only issue)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    # Try with weights_only=False for trusted model files
                    model_state = torch.load(model_path, map_location=device, weights_only=False)
                except Exception as e:
                    if "weights_only" in str(e):
                        print(f"  ⚠️ PyTorch 2.6 weights_only issue - trying with safe globals...")
                        # Add safe globals for omegaconf
                        torch.serialization.add_safe_globals([
                            'omegaconf.listconfig.ListConfig',
                            'omegaconf.dictconfig.DictConfig'
                        ])
                        model_state = torch.load(model_path, map_location=device, weights_only=True)
                    else:
                        raise e
                
                print(f"  ✅ {model_name} loaded successfully")
                print(f"  📊 Model keys: {len(model_state.keys()) if isinstance(model_state, dict) else 'Not a dict'}")
                
                # Show some key information
                if isinstance(model_state, dict):
                    if 'model' in model_state:
                        print(f"  🔑 Contains 'model' key")
                    if 'ema' in model_state:
                        print(f"  🔑 Contains 'ema' key (EMA weights)")
                    
                    # Show first few keys
                    keys = list(model_state.keys())[:5]
                    print(f"  🗝️  First keys: {keys}")
                
            except Exception as e:
                print(f"  ❌ {model_name} failed to load: {e}")
        else:
            print(f"❌ {model_name} not found")
    print()

def main():
    """Run all diagnostics."""
    print("🔍 HADM Model Diagnostics")
    print("=" * 80)
    print()
    
    check_python_environment()
    check_system_dependencies()
    check_basic_dependencies()
    check_detectron2()
    check_hadm_structure()
    check_model_files()
    check_hadm_imports()
    check_cuda()
    test_simple_model_loading()
    
    print("🎯 Diagnostic Summary")
    print("=" * 50)
    print("If you see ❌ errors above:")
    print("1. Install missing dependencies")
    print("2. Check HADM directory structure")
    print("3. Verify model files are downloaded")
    print("4. Check CUDA installation if using GPU")
    print()
    print("Common fixes:")
    print("- sudo apt-get install -y libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev  # System libs")
    print("- pip install --upgrade pip setuptools wheel")
    print("- pip install Pillow==9.0.0  # HADM compatibility fix")
    print("- pip install cloudpickle")
    print("- pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu128/torch2.0/index.html")
    print("- pip install opencv-python numpy")
    print("- Download missing model files")

if __name__ == "__main__":
    main() 