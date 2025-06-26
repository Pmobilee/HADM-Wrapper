#!/usr/bin/env python3
"""
HADM API Test Script
Tests the HADM Server API endpoints with a sample cat image or custom test image.
"""

import requests
import json
import time
import os
import sys
import argparse
from urllib.request import urlretrieve
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080/api/v1"
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1024px-Cat_November_2010-1a.jpg"
DEFAULT_TEST_IMAGE = "test_cat.jpg"
TEST_IMAGES_DIR = "tests/test_images"

def get_test_image_path(image_name=None):
    """Get the path to the test image."""
    if image_name:
        # Check if it's a custom image from test_images directory
        test_image_path = os.path.join(TEST_IMAGES_DIR, f"{image_name}.jpg")
        if os.path.exists(test_image_path):
            return test_image_path
        else:
            print(f"❌ Test image not found: {test_image_path}")
            # List available test images
            if os.path.exists(TEST_IMAGES_DIR):
                print(f"📁 Available test images in {TEST_IMAGES_DIR}:")
                for file in os.listdir(TEST_IMAGES_DIR):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        name_without_ext = os.path.splitext(file)[0]
                        print(f"   - {name_without_ext}")
            return None
    else:
        return DEFAULT_TEST_IMAGE

def download_test_image():
    """Download a test cat image."""
    print(f"📥 Downloading test image from: {TEST_IMAGE_URL}")
    try:
        urlretrieve(TEST_IMAGE_URL, DEFAULT_TEST_IMAGE)
        print(f"✅ Test image saved as: {DEFAULT_TEST_IMAGE}")
        return True
    except Exception as e:
        print(f"❌ Failed to download test image: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint."""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_info_endpoint():
    """Test the info endpoint."""
    print("\n📋 Testing info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Info endpoint working!")
            print(f"   Service: {data.get('name')} v{data.get('version')}")
            print(f"   Supported formats: {data.get('supported_formats')}")
            print(f"   Detection types: {data.get('detection_types')}")
            return True
        else:
            print(f"❌ Info endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Info endpoint error: {e}")
        return False

def test_detection_endpoint(endpoint_path, endpoint_name, test_image_path):
    """Test a detection endpoint."""
    print(f"\n🔬 Testing {endpoint_name} endpoint...")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': (os.path.basename(test_image_path), f, 'image/jpeg')}
            data = {
                'confidence_threshold': 0.3,
                'max_detections': 50
            }
            
            print(f"   Using image: {test_image_path}")
            print(f"   Sending request to: {API_BASE_URL}{endpoint_path}")
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}{endpoint_path}", files=files, data=data)
            end_time = time.time()
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {endpoint_name} detection successful!")
                print(f"   Success: {result.get('success')}")
                print(f"   Message: {result.get('message')}")
                print(f"   Processing Time: {result.get('processing_time', 0):.3f}s")
                print(f"   Image Size: {result.get('image_width')}x{result.get('image_height')}")
                
                # Show local detections if available
                local_detections = result.get('local_detections')
                if local_detections:
                    print(f"   Local Detections: {len(local_detections)} found")
                    for i, detection in enumerate(local_detections[:3]):  # Show first 3
                        bbox = detection.get('bbox', {})
                        print(f"     {i+1}. {detection.get('class_name')} "
                              f"(confidence: {detection.get('confidence', 0):.3f}) "
                              f"at [{bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}, "
                              f"{bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f}]")
                
                # Show global detection if available
                global_detection = result.get('global_detection')
                if global_detection:
                    print(f"   Global Detection: {global_detection.get('class_name')} "
                          f"(confidence: {global_detection.get('confidence', 0):.3f})")
                    
                    # Show top probabilities
                    probabilities = global_detection.get('probabilities', {})
                    if probabilities:
                        print("   Top Probabilities:")
                        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                        for class_name, prob in sorted_probs[:3]:
                            print(f"     {class_name}: {prob:.3f}")
                
                return True
            else:
                print(f"❌ {endpoint_name} detection failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ {endpoint_name} detection error: {e}")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test HADM API endpoints')
    parser.add_argument('--image', nargs='?', help='Name of test image (without .jpg extension) from tests/test_images/ directory')
    parser.add_argument('--list', action='store_true', help='List available test images')
    
    args = parser.parse_args()
    
    # List available images if requested
    if args.list:
        print("📁 Available test images:")
        if os.path.exists(TEST_IMAGES_DIR):
            for file in sorted(os.listdir(TEST_IMAGES_DIR)):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name_without_ext = os.path.splitext(file)[0]
                    print(f"   - {name_without_ext}")
        else:
            print(f"   No test images directory found: {TEST_IMAGES_DIR}")
        return
    
    print("🚀 HADM API Test Script")
    print("=" * 50)
    
    # Determine which image to use
    test_image_path = get_test_image_path(args.image)
    
    if not test_image_path:
        return
    
    # If using default image and it doesn't exist, download it
    if test_image_path == DEFAULT_TEST_IMAGE and not os.path.exists(test_image_path):
        if not download_test_image():
            print("❌ Cannot proceed without test image")
            return
    
    if args.image:
        print(f"✅ Using custom test image: {test_image_path}")
    else:
        print(f"✅ Using default test image: {test_image_path}")
    
    # Test endpoints
    results = []
    
    # Test health endpoint
    results.append(("Health", test_health_endpoint()))
    
    # Test info endpoint  
    results.append(("Info", test_info_endpoint()))
    
    # Test detection endpoints
    results.append(("Local Detection", test_detection_endpoint("/detect/local", "Local", test_image_path)))
    results.append(("Global Detection", test_detection_endpoint("/detect/global", "Global", test_image_path)))
    results.append(("Both Detection", test_detection_endpoint("/detect/both", "Both", test_image_path)))
    results.append(("Unified Detection", test_detection_endpoint("/detect", "Unified", test_image_path)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your HADM API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    # Cleanup (only remove default test image, not custom ones)
    if test_image_path == DEFAULT_TEST_IMAGE and os.path.exists(test_image_path):
        print(f"\n🧹 Cleaning up test image: {test_image_path}")
        os.remove(test_image_path)

if __name__ == "__main__":
    main() 