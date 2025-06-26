#!/usr/bin/env python3
"""
Test FastAPI app startup without running the server
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Testing FastAPI app startup...")

# Test 1: Test fallback setup
print("\n1. Setting up MMCV fallback...")
try:
    from app.utils.mmcv_fallback import setup_mmcv_fallback
    setup_mmcv_fallback()
    print("✅ MMCV fallback setup complete")
except Exception as e:
    print(f"❌ MMCV fallback setup failed: {e}")

# Test 2: Test app imports
print("\n2. Testing app imports...")
try:
    from app.core.config import settings
    print("✅ Settings imported")
    
    from app.core.hadm_models import model_manager
    print("✅ Model manager imported")
    
    from app.api.endpoints import router
    print("✅ API endpoints imported")
    
    from app.main import app
    print("✅ FastAPI app imported")
    
except Exception as e:
    print(f"❌ App import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test model manager initialization
print("\n3. Testing model manager...")
try:
    status = model_manager.get_model_status()
    print(f"✅ Model status: {status}")
    
    # Don't actually load models, just test the interface
    print("✅ Model manager interface working")
    
except Exception as e:
    print(f"❌ Model manager test failed: {e}")

# Test 4: Test app creation
print("\n4. Testing app object...")
try:
    print(f"✅ App title: {app.title}")
    print(f"✅ App version: {app.version}")
    print("✅ FastAPI app object created successfully")
    
except Exception as e:
    print(f"❌ App object test failed: {e}")

print("\n🎯 App startup test complete!")
print("If all tests pass, the app should start without import errors.") 