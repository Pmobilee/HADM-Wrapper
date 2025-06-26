#!/usr/bin/env python3
"""
Test the MMCV fallback system
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Testing MMCV fallback system...")

# Test 1: Import fallback module
print("\n1. Testing fallback module import...")
try:
    from app.utils.mmcv_fallback import setup_mmcv_fallback, soft_nms
    print("✅ Fallback module imported successfully")
    
    # Setup fallback
    result = setup_mmcv_fallback()
    print(f"✅ Fallback setup result: {result}")
    
except ImportError as e:
    print(f"❌ Fallback module import failed: {e}")
    sys.exit(1)

# Test 2: Test mmcv import after fallback
print("\n2. Testing mmcv import after fallback setup...")
try:
    import mmcv
    print(f"✅ MMCV imported: version {mmcv.__version__}")
    
    from mmcv import ops
    print("✅ MMCV ops imported")
    
    from mmcv.ops import soft_nms
    print("✅ MMCV soft_nms imported")
    
except ImportError as e:
    print(f"❌ MMCV import failed: {e}")

# Test 3: Test soft_nms functionality
print("\n3. Testing soft_nms functionality...")
try:
    import torch
    
    # Create test data
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [5, 5, 15, 15],
        [20, 20, 30, 30]
    ], dtype=torch.float32)
    
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    
    # Test soft_nms
    dets, keep = soft_nms(boxes, scores, iou_threshold=0.5)
    print(f"✅ Soft NMS result: {dets.shape[0]} detections kept")
    print(f"   Kept indices: {keep}")
    print(f"   Detection scores: {dets[:, 4]}")
    
except Exception as e:
    print(f"❌ Soft NMS test failed: {e}")

# Test 4: Test HADM import
print("\n4. Testing HADM imports with fallback...")
try:
    # Add HADM to path
    HADM_PATH = Path(__file__).parent / "HADM"
    if str(HADM_PATH) not in sys.path:
        sys.path.insert(0, str(HADM_PATH))
    
    from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
    print("✅ HADM fast_rcnn import successful with fallback")
    
except ImportError as e:
    print(f"❌ HADM fast_rcnn import failed: {e}")

print("\n🎯 Fallback system test complete!")
print("If all tests pass, the FastAPI app should work with the fallback system.") 