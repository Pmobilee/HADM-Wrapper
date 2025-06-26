#!/usr/bin/env python3
"""
HADM Predictor Direct Testing Script

This script tests the HADM models directly using the predictor.py from HADM/demo/
instead of going through our API. This helps us understand what the models
can actually do and compare with our API implementation.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Add HADM to Python path
HADM_PATH = Path(__file__).parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

# Import HADM components
from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg, LazyConfig
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

# Configuration
TEST_IMAGES_DIR = "tests/test_images"


def setup_hadm_local_config():
    """Setup configuration for HADM Local model using the actual HADM configs."""
    print("🔧 Setting up HADM Local configuration...")

    try:
        # Try to use the actual HADM config
        config_path = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
        if os.path.exists(config_path):
            print(f"✅ Found HADM config: {config_path}")
            cfg = LazyConfig.load_config(config_path)

            # Set device
            cfg.model.device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

            # Set model weights path
            model_paths = [
                "/home/pretrained_models/HADM-L_0249999.pth",
                "./pretrained_models/HADM-L_0249999.pth",
                "pretrained_models/HADM-L_0249999.pth",
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path:
                print(f"✅ Found model weights: {model_path}")
                cfg.model.weights = model_path
            else:
                print(f"❌ No model weights found in: {model_paths}")
                return None

            return cfg

        else:
            print(f"❌ HADM config not found: {config_path}")
            # Fallback to standard detectron2 config
            return setup_fallback_config("local")

    except Exception as e:
        print(f"❌ Error setting up HADM config: {e}")
        return setup_fallback_config("local")


def setup_hadm_global_config():
    """Setup configuration for HADM Global model."""
    print("🔧 Setting up HADM Global configuration...")

    try:
        # Try to use the actual HADM config
        config_path = "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
        if os.path.exists(config_path):
            print(f"✅ Found HADM config: {config_path}")
            cfg = LazyConfig.load_config(config_path)

            # Set device
            cfg.model.device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

            # Set model weights path
            model_paths = [
                "/home/pretrained_models/HADM-G_0249999.pth",
                "./pretrained_models/HADM-G_0249999.pth",
                "pretrained_models/HADM-G_0249999.pth",
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path:
                print(f"✅ Found model weights: {model_path}")
                cfg.model.weights = model_path
            else:
                print(f"❌ No model weights found in: {model_paths}")
                return None

            return cfg

        else:
            print(f"❌ HADM config not found: {config_path}")
            return setup_fallback_config("global")

    except Exception as e:
        print(f"❌ Error setting up HADM config: {e}")
        return setup_fallback_config("global")


def setup_fallback_config(model_type):
    """Setup fallback configuration using standard detectron2."""
    print(f"🔄 Setting up fallback config for {model_type}...")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    # Set device
    cfg.MODEL.DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

    # Set model weights
    model_name = f"HADM-{'L' if model_type == 'local' else 'G'}_0249999.pth"
    model_paths = [
        f"/home/pretrained_models/{model_name}",
        f"./pretrained_models/{model_name}",
        f"pretrained_models/{model_name}",
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path:
        print(f"✅ Found model weights: {model_path}")
        cfg.MODEL.WEIGHTS = model_path
    else:
        print(f"❌ No model weights found in: {model_paths}")
        return None

    # Adjust number of classes for HADM
    if model_type == "local":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # HADM-L classes
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # HADM-G classes

    return cfg


def test_predictor_on_image(image_path, model_type="local"):
    """Test the HADM predictor directly on an image."""
    print(f"\n🔬 Testing HADM {model_type.upper()} predictor on: {image_path}")
    print("-" * 60)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    try:
        # Setup configuration
        if model_type == "local":
            cfg = setup_hadm_local_config()
        else:
            cfg = setup_hadm_global_config()

        if cfg is None:
            print(f"❌ Failed to setup {model_type} configuration")
            return False

        # Create visualization demo
        print(f"🔄 Creating VisualizationDemo for {model_type}...")
        demo = VisualizationDemo(
            cfg,
            instance_mode=ColorMode.IMAGE,
            parallel=False,  # Keep it simple for testing
        )

        # Load image
        print(f"🖼️  Loading image: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False

        print(f"📸 Image shape: {image.shape}")

        # Run prediction
        print(f"🚀 Running {model_type} prediction...")
        start_time = time.time()

        predictions, vis_output = demo.run_on_image(image)

        inference_time = time.time() - start_time
        print(f"⏱️  Inference time: {inference_time:.3f}s")

        # Analyze predictions
        print(f"\n📊 Prediction Analysis:")
        print(f"   Prediction keys: {list(predictions.keys())}")

        if "instances" in predictions:
            instances = predictions["instances"]
            print(f"   Number of instances: {len(instances)}")

            if len(instances) > 0:
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()

                print(f"   Scores: {scores}")
                print(f"   Classes: {classes}")
                print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                print(f"   Mean score: {scores.mean():.3f}")

                # Show top detections
                if len(scores) > 0:
                    top_indices = np.argsort(scores)[::-1][:5]  # Top 5
                    print(f"\n🎯 Top detections:")
                    for i, idx in enumerate(top_indices):
                        box = boxes[idx]
                        print(
                            f"     {i+1}. Class {classes[idx]}, Score: {scores[idx]:.3f}, "
                            f"Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
                        )

                # Check for masks
                if hasattr(instances, "pred_masks"):
                    print(f"   Has segmentation masks: {instances.pred_masks.shape}")

                # Check for keypoints
                if hasattr(instances, "pred_keypoints"):
                    print(f"   Has keypoints: {instances.pred_keypoints.shape}")
            else:
                print("   ⚠️  No instances detected")

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            print(f"   Panoptic segmentation: {panoptic_seg.shape}")
            print(f"   Segments info: {len(segments_info)} segments")

        if "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"]
            print(f"   Semantic segmentation: {sem_seg.shape}")

        # Save visualization if available
        if vis_output is not None:
            output_path = f"output_{model_type}_{Path(image_path).stem}.jpg"
            vis_image = vis_output.get_image()
            cv2.imwrite(output_path, vis_image)
            print(f"💾 Saved visualization: {output_path}")

        # Memory usage
        try:
            import torch

            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(
                    f"🔥 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
                )
        except:
            pass

        return True

    except Exception as e:
        print(f"❌ Error testing {model_type} predictor: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False


def compare_with_api_results():
    """Compare results with our API implementation."""
    print("\n🔄 Comparing with API results...")
    print("=" * 60)

    # This would make API calls to compare
    # For now, just show what we should compare
    print("📋 Comparison checklist:")
    print("   ✓ Number of detections")
    print("   ✓ Confidence scores")
    print("   ✓ Bounding box coordinates")
    print("   ✓ Class predictions")
    print("   ✓ Processing time")
    print("   ✓ Memory usage")
    print("   ✓ Model architecture used")


def main():
    """Run comprehensive predictor testing."""
    print("🚀 HADM Predictor Direct Testing")
    print("=" * 60)

    # Check environment
    print("🔍 Environment Check:")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   HADM path: {HADM_PATH}")
    print(f"   HADM exists: {HADM_PATH.exists()}")

    # Find test images
    test_images = []
    if os.path.exists(TEST_IMAGES_DIR):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            test_images.extend(Path(TEST_IMAGES_DIR).glob(ext))

    if not test_images:
        # Look for any image files in current directory
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            test_images.extend(Path(".").glob(ext))

    if not test_images:
        print(f"❌ No test images found in {TEST_IMAGES_DIR} or current directory")
        print("   Please add some test images to proceed")
        return

    print(f"📸 Found {len(test_images)} test images")

    # Test both models on first image
    test_image = test_images[0]
    print(f"\n🖼️  Using test image: {test_image}")

    # Test local model
    success_local = test_predictor_on_image(test_image, "local")

    # Test global model
    success_global = test_predictor_on_image(test_image, "global")

    # Compare with API
    compare_with_api_results()

    # Summary
    print(f"\n✅ Testing Summary:")
    print("=" * 60)
    print(f"   Local model test: {'✅ Success' if success_local else '❌ Failed'}")
    print(f"   Global model test: {'✅ Success' if success_global else '❌ Failed'}")

    if success_local or success_global:
        print("\n🎯 Key Findings:")
        print("   - Check the confidence scores and number of detections")
        print("   - Compare memory usage with API implementation")
        print("   - Verify that proper HADM configs are being used")
        print("   - Look for differences in model architecture loading")
    else:
        print("\n❌ Both tests failed. Check:")
        print("   - Model file paths and availability")
        print("   - HADM directory structure")
        print("   - Dependencies and imports")


if __name__ == "__main__":
    main()
