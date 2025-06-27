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
import torch

# Setup mmcv fallback BEFORE any HADM imports
sys.path.insert(0, str(Path(__file__).parent / "app"))
from utils.mmcv_fallback import setup_mmcv_fallback

setup_mmcv_fallback()

# Add HADM to Python path
HADM_PATH = Path(__file__).parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

# Now safely import HADM components
try:
    from demo.predictor import VisualizationDemo
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import ColorMode

    # Try to import LazyConfig - this might not be available in older detectron2
    try:
        from detectron2.config import LazyConfig

        LAZY_CONFIG_AVAILABLE = True
    except ImportError:
        print("⚠️ LazyConfig not available - using standard config only")
        LAZY_CONFIG_AVAILABLE = False

except ImportError as e:
    print(f"❌ Failed to import HADM components: {e}")
    print("Make sure HADM directory exists and contains the required files")
    sys.exit(1)

# Configuration
TEST_IMAGES_DIR = "tests/test_images"


def setup_hadm_local_config():
    """Setup configuration for HADM Local model using the actual HADM configs."""
    print("🔧 Setting up HADM Local configuration...")

    try:
        if LAZY_CONFIG_AVAILABLE:
            # Correct path relative to the HADM root
            config_path = HADM_PATH / "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
            if config_path.exists():
                print(f"✅ Found HADM config: {config_path}")
                cfg = LazyConfig.load(str(config_path))

                # Set device and weights
                cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                model_paths = [
                    "/home/pretrained_models/HADM-L_0249999.pth",
                    "./pretrained_models/HADM-L_0249999.pth",
                    "pretrained_models/HADM-L_0249999.pth",
                ]
                
                model_path = next((p for p in model_paths if os.path.exists(p)), None)

                if model_path:
                    print(f"✅ Found model weights: {model_path}")
                    cfg.model.weights = model_path
                else:
                    print(f"❌ No model weights found in: {model_paths}")
                    return setup_fallback_config("local")
                
                return cfg
            else:
                print(f"❌ HADM config not found: {config_path}")
        else:
            print("⚠️ LazyConfig not available - using fallback config")

        return setup_fallback_config("local")

    except Exception as e:
        print(f"❌ Error setting up HADM config: {e}")
        return setup_fallback_config("local")


def setup_hadm_global_config():
    """Setup configuration for HADM Global model."""
    print("🔧 Setting up HADM Global configuration...")

    try:
        if LAZY_CONFIG_AVAILABLE:
            # Correct path relative to the HADM root
            config_path = HADM_PATH / "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
            if config_path.exists():
                print(f"✅ Found HADM config: {config_path}")
                cfg = LazyConfig.load(str(config_path))

                # Set device and weights
                cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"

                model_paths = [
                    "/home/pretrained_models/HADM-G_0249999.pth",
                    "./pretrained_models/HADM-G_0249999.pth",
                    "pretrained_models/HADM-G_0249999.pth",
                ]

                model_path = next((p for p in model_paths if os.path.exists(p)), None)

                if model_path:
                    print(f"✅ Found model weights: {model_path}")
                    cfg.model.weights = model_path
                else:
                    print(f"❌ No model weights found in: {model_paths}")
                    return setup_fallback_config("global")
                
                return cfg
            else:
                print(f"❌ HADM config not found: {config_path}")
        else:
            print("⚠️ LazyConfig not available - using fallback config")

        return setup_fallback_config("global")

    except Exception as e:
        print(f"❌ Error setting up HADM config: {e}")
        return setup_fallback_config("global")


def setup_fallback_config(model_type):
    """Setup fallback configuration using standard detectron2."""
    print(f"🔄 Setting up fallback config for {model_type}...")

    cfg = get_cfg()

    # Use a config that actually exists in the HADM detectron2 installation
    try:
        # Try to use HADM's own config files
        config_file = "HADM/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        if os.path.exists(config_file):
            print(f"✅ Using HADM config: {config_file}")
            cfg.merge_from_file(config_file)
        else:
            # Try without HADM prefix (if we're in the right directory)
            config_file = "configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            if os.path.exists(config_file):
                print(f"✅ Using config: {config_file}")
                cfg.merge_from_file(config_file)
            else:
                print("❌ No suitable config file found - using minimal config")
                # Create a minimal working config
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
                cfg.MODEL.RESNETS.DEPTH = 50
                cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
                cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
                cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
                cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
                cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    except Exception as e:
        print(f"⚠️ Error loading config file: {e}")
        print("Using minimal config")

    # Set device
    cfg.MODEL.DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

    # Add HADM-specific configurations that are missing
    print("🔧 Adding HADM-specific configurations...")

    # Add missing ROI_BOX_HEAD configurations
    if not hasattr(cfg.MODEL, "ROI_BOX_HEAD"):
        cfg.MODEL.ROI_BOX_HEAD = type(cfg.MODEL)()

    # Set all the missing attributes that HADM expects
    cfg.MODEL.ROI_BOX_HEAD.MULTI_LABEL = False  # This was the main missing attribute
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50

    # Add other potentially missing configurations
    if not hasattr(cfg, "DATASETS"):
        cfg.DATASETS = type(cfg)()
        cfg.DATASETS.TRAIN = ("coco_2017_train",)

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
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # HADM-L classes
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # HADM-G classes

    print(f"✅ Configuration setup complete for {model_type} model")
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
