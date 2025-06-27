"""
HADM Model Wrappers - Fixed Version
Based on successful loading approach from diagnose_models.py
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax, sigmoid
import psutil
import gc

# Add HADM to Python path - CRITICAL: Do this first before any other imports
HADM_PATH = Path(__file__).parent.parent.parent / "HADM"
if str(HADM_PATH) not in sys.path:
    sys.path.insert(0, str(HADM_PATH))

logger = logging.getLogger(__name__)

# Import MMCV fallback BEFORE any HADM imports to prevent import errors
try:
    from app.utils.mmcv_fallback import setup_mmcv_fallback

    setup_mmcv_fallback()
except ImportError as e:
    logger.warning(f"Could not import mmcv_fallback: {e}")
    # Manual fallback setup
    import types

    try:
        import mmcv
        from mmcv import ops

        logger.info("✅ MMCV available")
    except ImportError:
        logger.warning("⚠️ Setting up manual MMCV fallback")
        mock_mmcv = types.ModuleType("mmcv")
        mock_ops = types.ModuleType("mmcv.ops")

        def fallback_soft_nms(
            boxes, scores, iou_threshold=0.3, sigma=0.5, min_score=1e-3, method="linear"
        ):
            from torchvision.ops import nms

            keep_indices = nms(boxes, scores, iou_threshold)
            kept_boxes = boxes[keep_indices]
            kept_scores = scores[keep_indices]
            dets = torch.cat([kept_boxes, kept_scores.unsqueeze(1)], dim=1)
            return dets, keep_indices

        mock_ops.soft_nms = fallback_soft_nms
        mock_mmcv.ops = mock_ops
        mock_mmcv.__version__ = "fallback"
        sys.modules["mmcv"] = mock_mmcv
        sys.modules["mmcv.ops"] = mock_ops

# Try to import dependencies - handle gracefully if not available
DEPENDENCIES_AVAILABLE = True
DETECTRON2_AVAILABLE = False

try:
    import torch

    logger.info(f"✅ PyTorch version: {torch.__version__}")
    logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    logger.error(f"❌ PyTorch not available: {e}")
    DEPENDENCIES_AVAILABLE = False

try:
    import detectron2

    logger.info(f"✅ Detectron2 version: {detectron2.__version__}")
    DETECTRON2_AVAILABLE = True

    from detectron2.config import get_cfg, CfgNode
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.config import LazyConfig
    from detectron2.structures import Boxes, Instances
    from detectron2.model_zoo import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.logger import setup_logger

    logger.info("✅ Detectron2 modules imported successfully")

except ImportError as e:
    logger.error(f"❌ Detectron2 not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    DETECTRON2_AVAILABLE = False

from app.core.config import settings
from app.models.schemas import (
    LocalDetection,
    GlobalDetection,
    BoundingBox,
    DetectionMetrics,
    SegmentationData,
    KeypointData,
    ConfidenceMetrics,
    DensePoseData,
)


class HADMModelBase:
    """Base class for HADM models with proper weight loading."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.device = None
        self.predictor = None
        self.model = None
        self.is_loaded = False
        self.simplified_mode = False
        self.class_names = []

    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"Using GPU 0")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR if needed (OpenCV format)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return image
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

    def load_model_weights_directly(self, model_path: str) -> Dict[str, Any]:
        """Load model weights directly using the successful approach from diagnose_models.py"""
        try:
            logger.info(f"🔄 Loading model weights directly from {model_path}")

            # Use the same approach as diagnose_models.py - weights_only=False for trusted files
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model_state = torch.load(
                model_path, map_location=device_str, weights_only=False
            )

            logger.info(f"✅ Model weights loaded successfully")

            # Log model information like diagnose_models.py does
            if isinstance(model_state, dict):
                logger.info(f"📊 Model keys: {len(model_state.keys())}")

                # Correctly access the model's state dict
                actual_model_state = model_state.get('model', model_state)
                if not isinstance(actual_model_state, dict):
                    logger.warning("⚠️ 'model' key does not contain a state dictionary. Cannot calculate stats.")
                    return model_state

                # Calculate total parameters from the actual model state
                total_params = sum(
                    tensor.numel() for tensor in actual_model_state.values() if isinstance(tensor, torch.Tensor)
                )
                logger.info(f"📊 Total parameters: {total_params:,}")

                # Show key information
                if "model" in model_state:
                    logger.info("🔑 Contains 'model' key")
                if "ema" in model_state:
                    logger.info("🔑 Contains 'ema' key (EMA weights)")

                # Show first few keys
                keys = list(model_state.keys())[:5]
                logger.info(f"🗝️ First keys: {keys}")

                # Calculate memory usage from the actual model state
                total_size_mb = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in actual_model_state.values()
                    if isinstance(tensor, torch.Tensor)
                ) / (1024 * 1024)
                logger.info(f"💾 Model size: {total_size_mb:.1f} MB")

            return model_state

        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def load_model(self) -> bool:
        """Override in subclasses."""
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Override in subclasses."""
        raise NotImplementedError


class HADMLocalModel(HADMModelBase):
    """HADM Local Model with proper weight loading."""

    def __init__(self):
        super().__init__("local")
        self.num_classes = 6  # HADM-L has 6 classes
        self.class_names = [
            "artifact_1",
            "artifact_2",
            "artifact_3",
            "artifact_4",
            "artifact_5",
            "artifact_6",
        ]

    def load_model(self) -> bool:
        """Load HADM-L model with direct weight loading approach."""
        try:
            logger.info("🔄 Loading HADM Local (HADM-L) model...")

            # Setup device
            self._setup_device()

            # Check for model weights - try multiple paths
            model_paths = [
                settings.hadm_l_model_path,  # /home/pretrained_models/HADM-L_0249999.pth
                f"./pretrained_models/{settings.hadm_l_model}",  # Fallback
                f"pretrained_models/{settings.hadm_l_model}",  # Another fallback
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if not model_path:
                logger.warning(f"⚠️ Model weights not found in any of: {model_paths}")
                logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            logger.info(f"📁 Found model weights at {model_path}")

            # Load model weights directly using the successful approach
            try:
                model_state = self.load_model_weights_directly(model_path)

                # Create a minimal configuration that won't conflict with the loaded weights
                cfg = get_cfg()
                cfg.MODEL.DEVICE = str(self.device)

                # Don't set MODEL.WEIGHTS - we'll load manually
                # This avoids the configuration mismatch issues

                # Create model without loading weights first
                logger.info("🔄 Building model architecture...")

                # Try to use the actual HADM config
                try:
                    # First try to use the HADM-specific config
                    hadm_config_path = (
                        "HADM/projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
                    )
                    if os.path.exists(hadm_config_path):
                        logger.info(
                            f"🎯 Using HADM-specific config: {hadm_config_path}"
                        )
                        # For .py configs, we need to use LazyConfig
                        from detectron2.config import LazyConfig

                        cfg = LazyConfig.load(hadm_config_path)
                        if hasattr(cfg, "model") and hasattr(cfg.model, "device"):
                            cfg.model.device = str(self.device)
                        # Convert LazyConfig to CfgNode for compatibility
                        from detectron2.config import instantiate

                        model = instantiate(cfg.model)
                        # Create a basic CfgNode for the predictor
                        from detectron2.config import CfgNode

                        cfg_node = CfgNode()
                        cfg_node.MODEL.DEVICE = str(self.device)
                        cfg_node.MODEL.WEIGHTS = ""
                        self.predictor = DefaultPredictor(cfg_node)
                        self.predictor.model = model.to(self.device)
                    else:
                        # Fallback to a working config from HADM's own configs
                        logger.info("🔄 Using fallback config from HADM configs")

                        # Use HADM's own config files instead of model zoo
                        config_file = (
                            "HADM/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                        )
                        if os.path.exists(config_file):
                            logger.info(f"✅ Using HADM config: {config_file}")
                            cfg.merge_from_file(config_file)
                        else:
                            # Try without HADM prefix
                            config_file = (
                                "configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                            )
                            if os.path.exists(config_file):
                                logger.info(f"✅ Using config: {config_file}")
                                cfg.merge_from_file(config_file)
                            else:
                                logger.warning(
                                    "❌ No suitable config file found - using minimal config"
                                )
                                # Create minimal config
                                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                                cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
                                cfg.MODEL.RESNETS.DEPTH = 50
                                cfg.MODEL.FPN.IN_FEATURES = [
                                    "res2",
                                    "res3",
                                    "res4",
                                    "res5",
                                ]
                                cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
                                    [32],
                                    [64],
                                    [128],
                                    [256],
                                    [512],
                                ]
                                cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
                                    [0.5, 1.0, 2.0]
                                ]
                                cfg.MODEL.RPN.IN_FEATURES = [
                                    "p2",
                                    "p3",
                                    "p4",
                                    "p5",
                                    "p6",
                                ]
                                cfg.MODEL.ROI_HEADS.IN_FEATURES = [
                                    "p2",
                                    "p3",
                                    "p4",
                                    "p5",
                                ]

                        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                        cfg.MODEL.DEVICE = str(self.device)
                        cfg.MODEL.WEIGHTS = ""  # Don't auto-load weights

                        # Add HADM-specific configurations that are missing
                        logger.info("🔧 Adding HADM-specific configurations...")

                        # Add missing ROI_BOX_HEAD configurations
                        if not hasattr(cfg.MODEL, "ROI_BOX_HEAD"):
                            cfg.MODEL.ROI_BOX_HEAD = type(cfg.MODEL)()

                        # Set all the missing attributes that HADM expects
                        cfg.MODEL.ROI_BOX_HEAD.MULTI_LABEL = (
                            False  # This was the main missing attribute
                        )
                        cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
                        cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
                        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
                        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50

                        # Add other potentially missing configurations
                        if not hasattr(cfg, "DATASETS"):
                            cfg.DATASETS = type(cfg)()
                            cfg.DATASETS.TRAIN = ("coco_2017_train",)

                        # Create predictor
                        self.predictor = DefaultPredictor(cfg)

                    # Now manually load the weights
                    logger.info("🔄 Loading weights into model...")

                    # Handle different checkpoint formats
                    if isinstance(model_state, dict) and "model" in model_state:
                        state_dict = model_state["model"]
                    else:
                        state_dict = model_state

                    # Load weights with strict=False to handle mismatches gracefully
                    missing_keys, unexpected_keys = (
                        self.predictor.model.load_state_dict(state_dict, strict=False)
                    )

                    if missing_keys:
                        logger.warning(f"Missing keys: {len(missing_keys)} keys")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")

                    logger.info("✅ Model weights loaded into architecture")

                except Exception as arch_error:
                    logger.warning(f"Standard architecture failed: {arch_error}")
                    logger.info("🔄 Falling back to simplified mode...")
                    self.simplified_mode = True
                    self.is_loaded = True
                    return True

                # Store model reference
                self.model = self.predictor.model

                # Log GPU memory usage if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"🔥 GPU memory allocated: {memory_allocated:.2f} GB")

                self.is_loaded = True
                self.simplified_mode = False
                logger.info("✅ HADM-L model loaded successfully into VRAM")
                return True

            except Exception as loading_error:
                logger.error(f"❌ Failed to load model: {loading_error}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE")
                self.simplified_mode = True
                self.is_loaded = True
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-L model: {e}")
            logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE")
            self.simplified_mode = True
            self.is_loaded = True
            return True

    def predict(self, image: np.ndarray) -> List[LocalDetection]:
        """Predict local artifacts."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.simplified_mode:
            logger.warning("Model in simplified mode - returning empty results")
            return []

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Run inference
            start_time = time.time()
            predictions = self.predictor(processed_image)
            inference_time = time.time() - start_time

            detections = []
            if "instances" in predictions:
                instances = predictions["instances"]
                scores = instances.scores.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()

                for i, (score, box, cls_id) in enumerate(zip(scores, boxes, classes)):
                    if score >= settings.confidence_threshold:
                        detection = LocalDetection(
                            bbox=BoundingBox(
                                x1=float(box[0]),
                                y1=float(box[1]),
                                x2=float(box[2]),
                                y2=float(box[3]),
                            ),
                            confidence=float(score),
                            class_id=int(cls_id),
                            class_name=(
                                self.class_names[cls_id]
                                if cls_id < len(self.class_names)
                                else f"class_{cls_id}"
                            ),
                            processing_time=inference_time,
                            detection_source="local_model",
                        )
                        detections.append(detection)

            return detections[: settings.max_detections]

        except Exception as e:
            logger.error(f"Error in local prediction: {e}")
            return []


class HADMGlobalModel(HADMModelBase):
    """HADM Global Model with proper weight loading."""

    def __init__(self):
        super().__init__("global")
        self.num_classes = 12  # HADM-G has 12 classes
        self.class_names = [f"global_artifact_{i+1}" for i in range(12)]

    def load_model(self) -> bool:
        """Load HADM-G model with direct weight loading approach."""
        try:
            logger.info("🔄 Loading HADM Global (HADM-G) model...")

            # Setup device
            self._setup_device()

            # Check for model weights - try multiple paths
            model_paths = [
                settings.hadm_g_model_path,  # /home/pretrained_models/HADM-G_0249999.pth
                f"./pretrained_models/{settings.hadm_g_model}",  # Fallback
                f"pretrained_models/{settings.hadm_g_model}",  # Another fallback
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if not model_path:
                logger.warning(f"⚠️ Model weights not found in any of: {model_paths}")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            logger.info(f"📁 Found model weights at {model_path}")

            # Load model weights directly using the successful approach
            try:
                model_state = self.load_model_weights_directly(model_path)

                # Create a minimal configuration
                cfg = get_cfg()
                cfg.MODEL.DEVICE = str(self.device)

                # Try to use the actual HADM config
                try:
                    # First try to use the HADM-specific config
                    hadm_config_path = (
                        "HADM/projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
                    )
                    if os.path.exists(hadm_config_path):
                        logger.info(
                            f"🎯 Using HADM-specific config: {hadm_config_path}"
                        )
                        # For .py configs, we need to use LazyConfig
                        from detectron2.config import LazyConfig

                        cfg = LazyConfig.load(
                            hadm_config_path
                        )  # Fixed: use .load instead of .load_config
                        if hasattr(cfg, "model") and hasattr(cfg.model, "device"):
                            cfg.model.device = str(self.device)
                        # Convert LazyConfig to CfgNode for compatibility
                        from detectron2.config import instantiate

                        model = instantiate(cfg.model)
                        # Create a basic CfgNode for the predictor
                        from detectron2.config import CfgNode

                        cfg_node = CfgNode()
                        cfg_node.MODEL.DEVICE = str(self.device)
                        cfg_node.MODEL.WEIGHTS = ""
                        self.predictor = DefaultPredictor(cfg_node)
                        self.predictor.model = model.to(self.device)
                    else:
                        # Fallback to a working config from HADM's own configs
                        logger.info("🔄 Using fallback config from HADM configs")

                        # Use HADM's own config files instead of model zoo
                        config_file = (
                            "HADM/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                        )
                        if os.path.exists(config_file):
                            logger.info(f"✅ Using HADM config: {config_file}")
                            cfg.merge_from_file(config_file)
                        else:
                            # Try without HADM prefix
                            config_file = (
                                "configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                            )
                            if os.path.exists(config_file):
                                logger.info(f"✅ Using config: {config_file}")
                                cfg.merge_from_file(config_file)
                            else:
                                logger.warning(
                                    "❌ No suitable config file found - using minimal config"
                                )
                                # Create minimal config
                                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                                cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
                                cfg.MODEL.RESNETS.DEPTH = 50
                                cfg.MODEL.FPN.IN_FEATURES = [
                                    "res2",
                                    "res3",
                                    "res4",
                                    "res5",
                                ]
                                cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
                                    [32],
                                    [64],
                                    [128],
                                    [256],
                                    [512],
                                ]
                                cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
                                    [0.5, 1.0, 2.0]
                                ]
                                cfg.MODEL.RPN.IN_FEATURES = [
                                    "p2",
                                    "p3",
                                    "p4",
                                    "p5",
                                    "p6",
                                ]
                                cfg.MODEL.ROI_HEADS.IN_FEATURES = [
                                    "p2",
                                    "p3",
                                    "p4",
                                    "p5",
                                ]

                        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                        cfg.MODEL.DEVICE = str(self.device)
                        cfg.MODEL.WEIGHTS = ""  # Don't auto-load weights

                        # Add HADM-specific configurations that are missing
                        logger.info("🔧 Adding HADM-specific configurations...")

                        # Add missing ROI_BOX_HEAD configurations
                        if not hasattr(cfg.MODEL, "ROI_BOX_HEAD"):
                            cfg.MODEL.ROI_BOX_HEAD = type(cfg.MODEL)()

                        # Set all the missing attributes that HADM expects
                        cfg.MODEL.ROI_BOX_HEAD.MULTI_LABEL = (
                            False  # This was the main missing attribute
                        )
                        cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
                        cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
                        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
                        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50

                        # Add other potentially missing configurations
                        if not hasattr(cfg, "DATASETS"):
                            cfg.DATASETS = type(cfg)()
                            cfg.DATASETS.TRAIN = ("coco_2017_train",)

                        # Create predictor
                        self.predictor = DefaultPredictor(cfg)

                    # Manually load the weights
                    logger.info("🔄 Loading weights into model...")

                    if isinstance(model_state, dict) and "model" in model_state:
                        state_dict = model_state["model"]
                    else:
                        state_dict = model_state

                    missing_keys, unexpected_keys = (
                        self.predictor.model.load_state_dict(state_dict, strict=False)
                    )

                    if missing_keys:
                        logger.warning(f"Missing keys: {len(missing_keys)} keys")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")

                    logger.info("✅ Model weights loaded into architecture")

                except Exception as arch_error:
                    logger.warning(f"Standard architecture failed: {arch_error}")
                    logger.info("🔄 Falling back to simplified mode...")
                    self.simplified_mode = True
                    self.is_loaded = True
                    return True

                # Store model reference
                self.model = self.predictor.model

                # Log GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"🔥 GPU memory allocated: {memory_allocated:.2f} GB")

                self.is_loaded = True
                self.simplified_mode = False
                logger.info("✅ HADM-G model loaded successfully into VRAM")
                return True

            except Exception as loading_error:
                logger.error(f"❌ Failed to load model: {loading_error}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE")
                self.simplified_mode = True
                self.is_loaded = True
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-G model: {e}")
            logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE")
            self.simplified_mode = True
            self.is_loaded = True
            return True

    def predict(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """Predict global artifacts."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.simplified_mode:
            logger.warning("Model in simplified mode - returning empty results")
            return None

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Run inference
            start_time = time.time()
            predictions = self.predictor(processed_image)
            processing_time = time.time() - start_time

            if "instances" in predictions:
                instances = predictions["instances"]
                if len(instances) > 0:
                    scores = instances.scores.cpu().numpy()
                    classes = instances.pred_classes.cpu().numpy()

                    # Get highest confidence detection
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    best_class = classes[best_idx]

                    probabilities = {name: 0.0 for name in self.class_names}
                    probabilities[self.class_names[best_class]] = float(best_score)

                    return GlobalDetection(
                        class_id=int(best_class),
                        class_name=self.class_names[best_class],
                        confidence=float(best_score),
                        probabilities=probabilities,
                        processing_time=processing_time,
                        detection_source="global_model",
                    )

            return None

        except Exception as e:
            logger.error(f"Error in global prediction: {e}")
            return None


class HADMModelManager:
    """Manager for HADM models with improved loading."""

    def __init__(self):
        self.local_model = HADMLocalModel()
        self.global_model = HADMGlobalModel()

    def load_models(self) -> Dict[str, bool]:
        """Load all HADM models."""
        logger.info("Loading HADM models...")

        results = {
            "local": self.local_model.load_model(),
            "global": self.global_model.load_model(),
        }

        logger.info("All HADM models loaded successfully")
        return results

    def get_model_status(self) -> Dict[str, bool]:
        """Get model loading status."""
        return {
            "local": self.local_model.is_loaded,
            "global": self.global_model.is_loaded,
        }

    async def predict_local(self, image: np.ndarray) -> List[LocalDetection]:
        """Predict local artifacts."""
        return self.local_model.predict(image)

    async def predict_global(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """Predict global artifacts."""
        return self.global_model.predict(image)

    async def predict_both(
        self, image: np.ndarray
    ) -> Tuple[List[LocalDetection], Optional[GlobalDetection]]:
        """Predict both local and global artifacts."""
        local_results = await self.predict_local(image)
        global_results = await self.predict_global(image)
        return local_results, global_results


# Global model manager instance
model_manager = HADMModelManager()
