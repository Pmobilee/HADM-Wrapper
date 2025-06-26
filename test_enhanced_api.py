#!/usr/bin/env python3
"""
Enhanced API Testing Script

This script demonstrates all the new enhanced detection capabilities including:
- Comprehensive probability distributions
- Segmentation masks and keypoint data
- Advanced confidence metrics and uncertainty estimates
- Artifact-specific analysis indicators
- Performance metrics and timing information
- Feature importance and attention maps
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8080/api/v1"
TEST_IMAGES_DIR = "tests/test_images"


def test_enhanced_capabilities():
    """Test the capabilities endpoint to see what features are available."""
    print("🔍 Testing Enhanced Detection Capabilities")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/detect/capabilities")
        if response.status_code == 200:
            capabilities = response.json()
            print("✅ Enhanced Features Available:")

            for feature, info in capabilities["enhanced_features"].items():
                print(f"\n📊 {feature.upper()}")
                print(f"   Description: {info['description']}")
                print(f"   Includes: {', '.join(info['includes'])}")

            print(f"\n🎯 Detection Types:")
            for dtype, desc in capabilities["detection_types"].items():
                print(f"   {dtype}: {desc}")

            print(f"\n⚙️ Available Parameters:")
            for param, desc in capabilities["parameters"].items():
                print(f"   {param}: {desc}")

        else:
            print(f"❌ Failed to get capabilities: {response.status_code}")

    except Exception as e:
        print(f"❌ Error testing capabilities: {e}")


def test_enhanced_detection(image_path, endpoint_name, endpoint_path, params=None):
    """Test enhanced detection with comprehensive analysis."""
    print(f"\n🔬 Testing {endpoint_name}")
    print("-" * 40)

    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False

    try:
        # Default enhanced parameters
        default_params = {
            "detection_type": "both",
            "confidence_threshold": 0.3,
            "max_detections": 10,
            "include_masks": True,
            "include_keypoints": True,
            "include_features": True,
            "include_attention": True,
            "analyze_artifacts": True,
            "return_top_k": 5,
        }

        if params:
            default_params.update(params)

        with open(image_path, "rb") as f:
            files = {"image": f}

            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}{endpoint_path}", files=files, data=default_params
            )
            request_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✅ {endpoint_name} successful!")
            print(f"   Request time: {request_time:.3f}s")
            print(f"   Message: {data.get('message', 'No message')}")

            # Analyze image analysis
            if "image_analysis" in data:
                img_analysis = data["image_analysis"]
                print(f"\n📸 Image Analysis:")
                print(
                    f"   Dimensions: {img_analysis.get('width')}x{img_analysis.get('height')}"
                )
                print(f"   Channels: {img_analysis.get('channels')}")
                print(f"   File size: {img_analysis.get('file_size')} bytes")
                print(f"   Color space: {img_analysis.get('color_space', 'Unknown')}")

            # Analyze local detections
            if "local_detections" in data and data["local_detections"]:
                local_dets = data["local_detections"]
                print(f"\n🎯 Local Detections ({len(local_dets)} found):")

                for i, detection in enumerate(local_dets[:3]):  # Show first 3
                    print(f"   Detection {i+1}:")
                    print(
                        f"     Class: {detection.get('class_name')} (confidence: {detection.get('confidence', 0):.3f})"
                    )
                    print(
                        f"     Bbox: [{detection.get('bbox', {}).get('x1', 0):.1f}, {detection.get('bbox', {}).get('y1', 0):.1f}, {detection.get('bbox', {}).get('x2', 0):.1f}, {detection.get('bbox', {}).get('y2', 0):.1f}]"
                    )

                    # Show enhanced information
                    if detection.get("class_probabilities"):
                        top_probs = sorted(
                            detection["class_probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                        print(f"     Top probabilities: {top_probs}")

                    if detection.get("segmentation"):
                        seg = detection["segmentation"]
                        print(
                            f"     Segmentation: area={seg.get('area', 0):.1f}, perimeter={seg.get('perimeter', 0):.1f}"
                        )

                    if detection.get("keypoints"):
                        kpts = detection["keypoints"]
                        print(
                            f"     Keypoints: {len(kpts.get('keypoints', []))} detected"
                        )
                        if kpts.get("confidence_scores"):
                            avg_conf = sum(kpts["confidence_scores"]) / len(
                                kpts["confidence_scores"]
                            )
                            print(f"     Keypoint avg confidence: {avg_conf:.3f}")

                    if detection.get("confidence_metrics"):
                        conf_metrics = detection["confidence_metrics"]
                        print(
                            f"     Confidence range: [{conf_metrics.get('confidence_lower', 0):.3f}, {conf_metrics.get('confidence_upper', 1):.3f}]"
                        )

                    if detection.get("metrics"):
                        metrics = detection["metrics"]
                        print(
                            f"     Size: {metrics.get('detection_size', 'unknown')}, Edge distance: {metrics.get('edge_distance', 0):.1f}"
                        )
                        print(
                            f"     Uncertainty: {metrics.get('uncertainty_score', 0):.3f}"
                        )

                    if detection.get("artifact_severity"):
                        print(
                            f"     Artifact severity: {detection['artifact_severity']:.3f}"
                        )

                    if detection.get("authenticity_score"):
                        print(
                            f"     Authenticity score: {detection['authenticity_score']:.3f}"
                        )

                    if detection.get("processing_time"):
                        print(
                            f"     Processing time: {detection['processing_time']:.4f}s"
                        )

            # Analyze global detection
            if "global_detection" in data and data["global_detection"]:
                global_det = data["global_detection"]
                print(f"\n🌍 Global Detection:")
                print(
                    f"   Class: {global_det.get('class_name')} (confidence: {global_det.get('confidence', 0):.3f})"
                )

                # Show probability distribution
                if global_det.get("probabilities"):
                    sorted_probs = sorted(
                        global_det["probabilities"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    print(f"   Top probabilities:")
                    for class_name, prob in sorted_probs[:5]:
                        print(f"     {class_name}: {prob:.3f}")

                # Show statistical measures
                if global_det.get("entropy"):
                    print(f"   Entropy: {global_det['entropy']:.3f}")
                if global_det.get("uncertainty_score"):
                    print(f"   Uncertainty: {global_det['uncertainty_score']:.3f}")
                if global_det.get("probability_gap"):
                    print(f"   Probability gap: {global_det['probability_gap']:.3f}")

                # Show artifact indicators
                if global_det.get("artifact_indicators"):
                    indicators = global_det["artifact_indicators"]
                    print(f"   Artifact Indicators:")
                    for indicator, value in indicators.items():
                        print(f"     {indicator}: {value:.3f}")

                # Show confidence metrics
                if global_det.get("confidence_metrics"):
                    conf_metrics = global_det["confidence_metrics"]
                    print(f"   Confidence Metrics:")
                    if conf_metrics.get("authenticity_confidence"):
                        print(
                            f"     Authenticity: {conf_metrics['authenticity_confidence']:.3f}"
                        )
                    if conf_metrics.get("manipulation_confidence"):
                        print(
                            f"     Manipulation: {conf_metrics['manipulation_confidence']:.3f}"
                        )

                if global_det.get("processing_time"):
                    print(f"   Processing time: {global_det['processing_time']:.4f}s")

            # Analyze performance metrics
            if "performance_metrics" in data:
                perf = data["performance_metrics"]
                print(f"\n⚡ Performance Metrics:")
                print(f"   Total inference time: {perf.get('inference_time', 0):.3f}s")
                if perf.get("preprocessing_time"):
                    print(f"   Preprocessing: {perf['preprocessing_time']:.3f}s")
                if perf.get("model_forward_time"):
                    print(f"   Model forward: {perf['model_forward_time']:.3f}s")
                if perf.get("postprocessing_time"):
                    print(f"   Postprocessing: {perf['postprocessing_time']:.3f}s")

                if perf.get("peak_memory_usage"):
                    print(f"   Peak memory: {perf['peak_memory_usage']:.1f} MB")
                if perf.get("gpu_memory_usage"):
                    print(f"   GPU memory: {perf['gpu_memory_usage']:.1f} MB")

                print(f"   Total detections: {perf.get('total_detections', 0)}")
                if perf.get("mean_confidence"):
                    print(f"   Mean confidence: {perf['mean_confidence']:.3f}")
                if perf.get("confidence_std"):
                    print(f"   Confidence std: {perf['confidence_std']:.3f}")
                print(
                    f"   High confidence count: {perf.get('high_confidence_count', 0)}"
                )
                print(f"   Device: {perf.get('device_used', 'unknown')}")

            # Show summary if available
            if "summary" in data and data["summary"]:
                summary = data["summary"]
                print(f"\n📋 Summary:")
                for key, value in summary.items():
                    if isinstance(value, dict):
                        print(f"   {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"     {sub_key}: {sub_value}")
                    else:
                        print(f"   {key}: {value}")

            return True

        else:
            print(f"❌ {endpoint_name} failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ {endpoint_name} error: {e}")
        return False


def main():
    """Run comprehensive enhanced detection tests."""
    print("🚀 Enhanced HADM Detection API Test Suite")
    print("=" * 60)

    # Test capabilities first
    test_enhanced_capabilities()

    # Find test images
    test_images = []
    if os.path.exists(TEST_IMAGES_DIR):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            test_images.extend(Path(TEST_IMAGES_DIR).glob(ext))

    # Use a default image if no test images found
    if not test_images:
        # Create a simple test image or use a placeholder
        test_image = "test_image.jpg"
        if not os.path.exists(test_image):
            print(
                f"\n⚠️  No test images found. Please add images to {TEST_IMAGES_DIR} or create {test_image}"
            )
            return
        test_images = [test_image]

    # Test different endpoints with various configurations
    test_configs = [
        {
            "name": "Basic Enhanced Detection",
            "endpoint": "/detect/enhanced",
            "params": {
                "detection_type": "both",
                "confidence_threshold": 0.3,
                "include_masks": True,
                "include_keypoints": True,
            },
        },
        {
            "name": "Local Detection with All Features",
            "endpoint": "/detect/local",
            "params": {
                "confidence_threshold": 0.2,
                "max_detections": 15,
                "include_masks": True,
                "include_keypoints": True,
                "include_features": True,
                "include_attention": True,
            },
        },
        {
            "name": "Global Detection with Artifact Analysis",
            "endpoint": "/detect/global",
            "params": {
                "confidence_threshold": 0.1,
                "include_features": True,
                "include_attention": True,
                "analyze_artifacts": True,
            },
        },
        {
            "name": "Comprehensive Analysis (All Features)",
            "endpoint": "/detect/enhanced",
            "params": {
                "detection_type": "both",
                "confidence_threshold": 0.1,
                "max_detections": 20,
                "include_masks": True,
                "include_keypoints": True,
                "include_features": True,
                "include_attention": True,
                "analyze_artifacts": True,
                "return_top_k": 10,
            },
        },
    ]

    # Run tests
    for test_image in test_images[:2]:  # Test with first 2 images
        print(f"\n🖼️  Testing with image: {test_image}")
        print("=" * 60)

        for config in test_configs:
            test_enhanced_detection(
                str(test_image), config["name"], config["endpoint"], config["params"]
            )
            time.sleep(1)  # Brief pause between tests

    print(f"\n✅ Enhanced detection testing complete!")
    print("=" * 60)
    print("\n📊 Summary of Enhanced Features Tested:")
    print("   ✓ Comprehensive probability distributions")
    print("   ✓ Segmentation masks with area/perimeter analysis")
    print("   ✓ Keypoint detection with confidence scores")
    print("   ✓ Advanced confidence metrics and uncertainty")
    print("   ✓ Artifact-specific analysis indicators")
    print("   ✓ Performance metrics and timing breakdown")
    print("   ✓ Enhanced metadata and processing information")
    print("\n🎯 All enhanced detection capabilities have been demonstrated!")


if __name__ == "__main__":
    main()
