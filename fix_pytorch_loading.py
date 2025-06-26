#!/usr/bin/env python3
"""
Fix PyTorch 2.6 weights_only issue in HADM model loading.
This script patches the torch.load calls to handle the new PyTorch 2.6 behavior.
"""

import re
import sys
import torch

def fix_pytorch_loading():
    """Add safe globals for omegaconf to handle PyTorch 2.6 weights_only issue."""
    try:
        # Add safe globals for omegaconf types
        torch.serialization.add_safe_globals([
            'omegaconf.listconfig.ListConfig',
            'omegaconf.dictconfig.DictConfig',
            'omegaconf.basecontainer.BaseContainer',
            'omegaconf.nodes.AnyNode',
            'omegaconf.nodes.StringNode',
            'omegaconf.nodes.IntegerNode',
            'omegaconf.nodes.FloatNode',
            'omegaconf.nodes.BooleanNode',
        ])
        print("✅ Added safe globals for omegaconf types")
        return True
    except Exception as e:
        print(f"❌ Failed to add safe globals: {e}")
        return False

def patch_file(filepath, old_pattern, new_code):
    """Patch a file by replacing a pattern with new code."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace the pattern
        new_content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE | re.DOTALL)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"✅ Patched {filepath}")
            return True
        else:
            print(f"⚠️ No changes needed in {filepath}")
            return False
    except Exception as e:
        print(f"❌ Failed to patch {filepath}: {e}")
        return False

def main():
    """Main function to fix PyTorch loading issues."""
    print("🔧 Fixing PyTorch 2.6 weights_only issues...")
    
    # Fix the global torch settings
    fix_pytorch_loading()
    
    # Pattern to match torch.load calls
    torch_load_pattern = r'torch\.load\(([^,]+),\s*map_location=([^)]+)\)'
    
    # Replacement with weights_only=False for trusted model files
    torch_load_replacement = r'''try:
                self.model_state = torch.load(\1, map_location=\2, weights_only=False)
            except Exception as e:
                if "weights_only" in str(e):
                    logger.info("PyTorch 2.6 weights_only issue - using safe globals...")
                    torch.serialization.add_safe_globals([
                        'omegaconf.listconfig.ListConfig',
                        'omegaconf.dictconfig.DictConfig'
                    ])
                    self.model_state = torch.load(\1, map_location=\2, weights_only=True)
                else:
                    raise e'''
    
    # Files to patch
    files_to_patch = [
        'app/core/hadm_models.py',
        'diagnose_models.py'
    ]
    
    for filepath in files_to_patch:
        try:
            # Read file
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Replace simple torch.load calls with error handling
            if 'torch.load(' in content and 'weights_only' not in content:
                # Replace torch.load with weights_only=False
                content = re.sub(
                    r'torch\.load\(([^,]+),\s*map_location=([^)]+)\)',
                    r'torch.load(\1, map_location=\2, weights_only=False)',
                    content
                )
                
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"✅ Added weights_only=False to {filepath}")
        except Exception as e:
            print(f"❌ Failed to patch {filepath}: {e}")
    
    print("\n✅ PyTorch loading fixes applied!")
    print("🧪 Test with: python diagnose_models.py")

if __name__ == "__main__":
    main() 