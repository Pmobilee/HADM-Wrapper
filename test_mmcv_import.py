#!/usr/bin/env python3
"""
Test script to isolate the mmcv import issue
"""

import sys
from pathlib import Path

# Add HADM to Python path
HADM_PATH = Path(__file__).parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

print("🔍 Testing imports step by step...")

print("\n1. Testing basic imports...")
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import detectron2
    print(f"✅ Detectron2: {detectron2.__version__}")
except ImportError as e:
    print(f"❌ Detectron2: {e}")

print("\n2. Testing mmcv...")
try:
    import mmcv
    print(f"✅ MMCV: {mmcv.__version__}")
    
    try:
        from mmcv import ops
        print("✅ MMCV ops available")
        
        try:
            from mmcv.ops import soft_nms
            print("✅ MMCV soft_nms available")
        except ImportError as e:
            print(f"❌ MMCV soft_nms: {e}")
            
    except ImportError as e:
        print(f"❌ MMCV ops: {e}")
        
except ImportError as e:
    print(f"❌ MMCV: {e}")

print("\n3. Testing HADM detectron2 import...")
try:
    from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
    print("✅ HADM fast_rcnn import successful")
except ImportError as e:
    print(f"❌ HADM fast_rcnn import: {e}")

print("\n4. Testing HADM config imports...")
try:
    from projects.ViTDet.configs.eva2_o365_to_coco.demo_local import model as local_model_config
    print("✅ HADM local config import successful")
except ImportError as e:
    print(f"❌ HADM local config import: {e}")

try:
    from projects.ViTDet.configs.eva2_o365_to_coco.demo_global import model as global_model_config
    print("✅ HADM global config import successful")
except ImportError as e:
    print(f"❌ HADM global config import: {e}")

print("\n5. Testing direct model loading (like diagnose_models.py)...")
try:
    model_path = Path("pretrained_models/HADM-L_0249999.pth")
    if model_path.exists():
        model_state = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"✅ Direct model loading successful: {len(model_state.keys()) if isinstance(model_state, dict) else 'Not a dict'} keys")
    else:
        print(f"❌ Model file not found: {model_path}")
except Exception as e:
    print(f"❌ Direct model loading: {e}")

print("\n🎯 Summary:")
print("If steps 1-2 work but step 3-4 fail, the issue is HADM's detectron2 trying to import mmcv.ops")
print("If step 5 works, it confirms the models can be loaded without the HADM config system") 