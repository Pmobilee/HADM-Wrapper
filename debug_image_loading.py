#!/usr/bin/env python3
"""
Debug script to test image loading issues
"""

import io
import requests
from PIL import Image
import numpy as np
import cv2
from urllib.request import urlretrieve

# Same image as test script
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1024px-Cat_November_2010-1a.jpg"
TEST_IMAGE_PATH = "debug_cat.jpg"

def test_direct_loading():
    """Test loading image directly from file."""
    print("🔍 Testing direct file loading...")
    
    # Download image
    print(f"📥 Downloading: {TEST_IMAGE_URL}")
    urlretrieve(TEST_IMAGE_URL, TEST_IMAGE_PATH)
    
    # Load with PIL
    try:
        pil_image = Image.open(TEST_IMAGE_PATH)
        print(f"✅ PIL loaded: {pil_image.size}, mode: {pil_image.mode}")
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy
        image_array = np.array(pil_image)
        print(f"✅ NumPy array: {image_array.shape}")
        
        # Convert to BGR
        bgr_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        print(f"✅ BGR array: {bgr_array.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        return False

def test_bytesio_loading():
    """Test loading image from BytesIO (like the API does)."""
    print("\n🔍 Testing BytesIO loading...")
    
    try:
        # Read file as bytes
        with open(TEST_IMAGE_PATH, 'rb') as f:
            content = f.read()
        
        print(f"📄 File size: {len(content)} bytes")
        
        # Create BytesIO
        image_bytes = io.BytesIO(content)
        image_bytes.seek(0)
        
        # Load with PIL
        pil_image = Image.open(image_bytes)
        print(f"✅ PIL from BytesIO: {pil_image.size}, mode: {pil_image.mode}")
        
        # Test verify
        image_bytes.seek(0)
        test_image = Image.open(image_bytes)
        test_image.verify()
        print("✅ Image verification passed")
        
        # Reopen for processing
        image_bytes.seek(0)
        pil_image = Image.open(image_bytes)
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy
        image_array = np.array(pil_image)
        print(f"✅ NumPy from BytesIO: {image_array.shape}")
        
        return True
    except Exception as e:
        print(f"❌ BytesIO loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_simulation():
    """Simulate the API upload process."""
    print("\n🔍 Testing API simulation...")
    
    try:
        # Simulate file upload
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'image': (TEST_IMAGE_PATH, f, 'image/jpeg')}
            
            # This simulates what requests does
            file_content = f.read()
            f.seek(0)  # Reset for actual upload
            
            print(f"📄 Upload content size: {len(file_content)} bytes")
            
            # Test BytesIO with this content
            image_bytes = io.BytesIO(file_content)
            image_bytes.seek(0)
            
            pil_image = Image.open(image_bytes)
            print(f"✅ API simulation: {pil_image.size}, mode: {pil_image.mode}")
            
            return True
    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Image Loading Debug Script")
    print("=" * 40)
    
    results = []
    results.append(("Direct Loading", test_direct_loading()))
    results.append(("BytesIO Loading", test_bytesio_loading()))
    results.append(("API Simulation", test_api_simulation()))
    
    print("\n" + "=" * 40)
    print("📊 Results:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    # Cleanup
    import os
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)
        print(f"\n🧹 Cleaned up: {TEST_IMAGE_PATH}")

if __name__ == "__main__":
    main() 