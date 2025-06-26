#!/usr/bin/env python3
"""
Test script to check HADM model loading
"""
import os
import sys
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_model_loading():
    """Test if HADM models can load properly."""
    try:
        from app.core.hadm_models import HADMLocalModel, HADMGlobalModel
        from app.core.config import settings
        
        print("🔄 Testing HADM model loading...")
        print(f"📁 Model path: {settings.model_path}")
        print(f"📁 HADM-L model: {settings.hadm_l_model_path}")
        print(f"📁 HADM-G model: {settings.hadm_g_model_path}")
        
        # Check if model files exist
        print(f"📋 HADM-L exists: {os.path.exists(settings.hadm_l_model_path)}")
        print(f"📋 HADM-G exists: {os.path.exists(settings.hadm_g_model_path)}")
        
        # Test local model
        print("\n🔄 Testing HADM-L loading...")
        local_model = HADMLocalModel()
        local_success = local_model.load_model()
        print(f"📋 HADM-L loaded: {local_success}")
        print(f"📋 HADM-L simplified mode: {getattr(local_model, 'simplified_mode', False)}")
        
        # Test global model  
        print("\n🔄 Testing HADM-G loading...")
        global_model = HADMGlobalModel()
        global_success = global_model.load_model()
        print(f"📋 HADM-G loaded: {global_success}")
        print(f"📋 HADM-G simplified mode: {getattr(global_model, 'simplified_mode', False)}")
        
        # Check GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                print(f"🔥 GPU memory allocated: {memory_allocated:.2f} GB")
            else:
                print("⚠️ CUDA not available")
        except Exception as e:
            print(f"⚠️ Could not check GPU memory: {e}")
        
        return local_success and global_success
        
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print(f"\n{'✅' if success else '❌'} Model loading test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1) 