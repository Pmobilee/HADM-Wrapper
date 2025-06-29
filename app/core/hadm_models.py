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
    ModelStatus,
    GPUInfo,
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
        self.model_path: Optional[str] = None
        self.model_size_mb: Optional[float] = None

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
            self.model_path = model_path

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
                self.model_size_mb = round(total_size_mb, 1)

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
            
            self.model_path = model_path
            logger.info(f"Found HADM-L model at: {self.model_path}")

            # Use LazyConfig for model configuration
            cfg = LazyConfig.load_file(f"{str(HADM_PATH)}/configs/HADM/HADM_L.py")

            # Build the model
            self.model = build_model(cfg.model)
            self.model.to(self.device)
            self.model.eval()

            # Load weights directly
            model_state = self.load_model_weights_directly(self.model_path)
            
            # Use DetectionCheckpointer to load state
            checkpointer = DetectionCheckpointer(self.model)
            
            # Try to load 'model' key, fall back to the whole dict
            if 'model' in model_state:
                checkpointer.load(self.model_path) # Let checkpointer handle it
                logger.info("Loaded weights using DetectionCheckpointer from 'model' key.")
            else:
                # This path may be needed if weights are not in 'model' sub-dict
                self.model.load_state_dict(model_state, strict=False)
                logger.info("Loaded weights directly using model.load_state_dict().")


            self.predictor = DefaultPredictor(cfg)
            
            # We need to manually set the model weights for the predictor if using direct load
            self.predictor.model.load_state_dict(self.model.state_dict())


            self.is_loaded = True
            logger.info("✅ HADM-L model loaded successfully")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-L model: {e}")
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False

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
            self._setup_device()

            # Check for model weights
            model_paths = [
                settings.hadm_g_model_path,
                f"./pretrained_models/{settings.hadm_g_model}",
                f"pretrained_models/{settings.hadm_g_model}",
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                logger.warning(f"⚠️ Model weights not found for HADM-G in: {model_paths}")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            self.model_path = model_path
            logger.info(f"Found HADM-G model at: {self.model_path}")
            
            # Configuration
            cfg_path = os.path.join(HADM_PATH, "configs/HADM/HADM_G.py")
            cfg = LazyConfig.load_file(cfg_path)
            
            # Build model
            self.model = build_model(cfg.model)
            self.model.to(self.device)
            self.model.eval()
            
            # Load weights
            model_state = self.load_model_weights_directly(self.model_path)
            checkpointer = DetectionCheckpointer(self.model)
            
            if 'model' in model_state:
                checkpointer.load(self.model_path)
                logger.info("Loaded weights using DetectionCheckpointer from 'model' key for HADM-G.")
            else:
                self.model.load_state_dict(model_state, strict=False)
                logger.info("Loaded weights directly for HADM-G.")

            # Create predictor
            self.predictor = DefaultPredictor(cfg)
            self.predictor.model.load_state_dict(self.model.state_dict())
            
            self.is_loaded = True
            logger.info("✅ HADM-G model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-G model: {e}")
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False

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
    """Singleton manager for HADM models."""

    def __init__(self):
        self.local_model = HADMLocalModel()
        self.global_model = HADMGlobalModel()
        self._is_loading = False

    def load_models(self) -> Dict[str, bool]:
        """Load all HADM models."""
        if self._is_loading:
            logger.warning("Models are already in the process of loading.")
            return {"status": "loading"}

        self._is_loading = True
        status = {}
        try:
            logger.info("--- Loading HADM Local Model ---")
            status["local_model_loaded"] = self.local_model.load_model()
            logger.info("--- Loading HADM Global Model ---")
            status["global_model_loaded"] = self.global_model.load_model()
        finally:
            self._is_loading = False
        return status

    def get_model_status(self) -> Dict[str, ModelStatus]:
        """Get the status of all managed models."""
        return {
            "local_model": ModelStatus(
                is_loaded=self.local_model.is_loaded,
                model_path=self.local_model.model_path,
                model_size_mb=self.local_model.model_size_mb,
                simplified_mode=self.local_model.simplified_mode,
            ),
            "global_model": ModelStatus(
                is_loaded=self.global_model.is_loaded,
                model_path=self.global_model.model_path,
                model_size_mb=self.global_model.model_size_mb,
                simplified_mode=self.global_model.simplified_mode,
            ),
        }

    def get_gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU information if available."""
        if not torch.cuda.is_available():
            return None

        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory, _ = torch.cuda.mem_get_info(device)
        used_memory = total_memory - free_memory
        
        # More detailed stats
        mem_stats = torch.cuda.memory_stats(device)
        
        return GPUInfo(
            device_name=torch.cuda.get_device_name(device),
            total_memory_mb=round(total_memory / (1024 * 1024), 2),
            used_memory_mb=round(used_memory / (1024 * 1024), 2),
            free_memory_mb=round(free_memory / (1024 * 1024), 2),
            active_memory_mb=round(mem_stats.get("active_bytes.all.current", 0) / (1024 * 1024), 2),
            allocated_memory_mb=round(mem_stats.get("allocated_bytes.all.current", 0) / (1024 * 1024), 2),
            reserved_memory_mb=round(mem_stats.get("reserved_bytes.all.current", 0) / (1024 * 1024), 2),
        )

    async def predict_local(self, image: np.ndarray) -> List[LocalDetection]:
        """Run prediction with the local model."""
        return self.local_model.predict(image)

    async def predict_global(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """Run prediction with the global model."""
        return self.global_model.predict(image)

    async def predict_both(
        self, image: np.ndarray
    ) -> Tuple[List[LocalDetection], Optional[GlobalDetection]]:
        """Run prediction with both models."""
        local_results = await self.predict_local(image)
        global_result = await self.predict_global(image)
        return local_results, global_result


# Global model manager instance
model_manager = HADMModelManager()
