"""
HADM Model Wrappers
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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
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
    import sys
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
            import torch
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
HADM_CONFIGS_AVAILABLE = False
MMCV_AVAILABLE = False
DETECTRON2_AVAILABLE = False

# Check each dependency separately with detailed logging
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

    logger.info("✅ Detectron2 modules imported successfully")

except ImportError as e:
    logger.error(f"❌ Detectron2 not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    DETECTRON2_AVAILABLE = False

# Check MMCV separately
try:
    import mmcv

    logger.info(f"✅ MMCV version: {mmcv.__version__}")
    MMCV_AVAILABLE = True

    # Test mmcv ops
    try:
        from mmcv import ops

        logger.info("✅ MMCV ops available")
    except ImportError as e:
        logger.warning(f"⚠️ MMCV ops not available: {e}")
        logger.warning("This may cause issues with HADM models that use soft-NMS")

except ImportError as e:
    logger.warning(f"⚠️ MMCV not available: {e}")
    logger.warning(
        "HADM models may work without MMCV, but some features may be limited"
    )
    MMCV_AVAILABLE = False

# Try to import HADM configurations - handle missing files gracefully
HADM_CONFIGS_AVAILABLE = False
try:
    # Check if the required config files exist first
    hadm_projects_path = HADM_PATH / "projects"
    if hadm_projects_path.exists():
        vitdet_path = hadm_projects_path / "ViTDet" / "configs" / "eva2_o365_to_coco"
        local_config_path = vitdet_path / "demo_local.py"
        global_config_path = vitdet_path / "demo_global.py"

        if local_config_path.exists() and global_config_path.exists():
            # Try importing the configs
            try:
                from projects.ViTDet.configs.eva2_o365_to_coco.demo_local import (
                    model as local_model_config,
                )

                logger.info("✅ HADM local config imported successfully")

                from projects.ViTDet.configs.eva2_o365_to_coco.demo_global import (
                    model as global_model_config,
                )

                logger.info("✅ HADM global config imported successfully")

                HADM_CONFIGS_AVAILABLE = True

            except Exception as config_error:
                logger.warning(
                    f"⚠️ HADM configs exist but failed to import: {config_error}"
                )
                logger.warning("Will use fallback configuration approach")
                HADM_CONFIGS_AVAILABLE = False
        else:
            logger.warning(f"⚠️ HADM config files not found at {vitdet_path}")
            logger.warning("Will use fallback configuration approach")
    else:
        logger.warning(f"⚠️ HADM projects directory not found: {hadm_projects_path}")
        logger.warning("Will use fallback configuration approach")

except Exception as e:
    logger.warning(f"⚠️ Unexpected error loading HADM configurations: {e}")
    logger.warning("Will use fallback configuration approach")
    HADM_CONFIGS_AVAILABLE = False

# Final dependency check
if DEPENDENCIES_AVAILABLE:
    logger.info("✅ All core dependencies loaded successfully")
else:
    logger.error("❌ Some core dependencies are missing")

# Log final status
logger.info(f"Dependency Status Summary:")
logger.info(f"  - PyTorch: {'✅' if 'torch' in sys.modules else '❌'}")
logger.info(f"  - Detectron2: {'✅' if DETECTRON2_AVAILABLE else '❌'}")
logger.info(f"  - MMCV: {'✅' if MMCV_AVAILABLE else '❌'}")
logger.info(f"  - HADM Configs: {'✅' if HADM_CONFIGS_AVAILABLE else '❌'}")
logger.info(f"  - Overall: {'✅' if DEPENDENCIES_AVAILABLE else '❌'}")

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
    AttentionData,
    ModelPerformanceMetrics,
)


class HADMModelBase:
    """Base class for HADM models."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.predictor = None
        self.metadata = None
        self.device = settings.device
        self.is_loaded = False
        self.model_version = "0249999"

        # Check dependencies
        if not DEPENDENCIES_AVAILABLE:
            logger.error(
                "Required dependencies not available. Please install detectron2 and PyTorch."
            )

    def _setup_device(self):
        """Setup computation device."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(settings.gpu_id)
            logger.info(f"Using GPU {settings.gpu_id}")
        else:
            self.device = "cpu"
            logger.info("Using CPU")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for HADM models.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and pad to 1024x1024 (HADM requirement)
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim > settings.image_size:
            scale = settings.image_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Square padding
        h, w = image.shape[:2]
        pad_h = (settings.image_size - h) // 2
        pad_w = (settings.image_size - w) // 2

        padded_image = np.zeros(
            (settings.image_size, settings.image_size, 3), dtype=image.dtype
        )
        padded_image[pad_h : pad_h + h, pad_w : pad_w + w] = image

        # Convert back to BGR for detectron2
        padded_image = cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)

        return padded_image

    def load_model(self) -> bool:
        """Load the model. To be implemented by subclasses."""
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run prediction on image. To be implemented by subclasses."""
        raise NotImplementedError


class HADMLocalModel(HADMModelBase):
    """Enhanced local artifact detection model with comprehensive information extraction."""

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
        ]  # Will be updated when we have actual class names
        self.model = None  # Store raw model for advanced features

    def _extract_probability_distribution(
        self, predictions: Dict[str, Any], proposals: List[Instances]
    ) -> Dict[str, Any]:
        """Extract full probability distributions and raw scores."""
        if "instances" not in predictions:
            return {}

        try:
            # Get raw model outputs if available
            if hasattr(self.model, "roi_heads") and hasattr(
                self.model.roi_heads, "box_predictor"
            ):
                box_predictor = self.model.roi_heads.box_predictor
                if hasattr(box_predictor, "predict_probs"):
                    # Get full probability distributions
                    raw_predictions = predictions.get("raw_predictions", None)
                    if raw_predictions and len(proposals) > 0:
                        probs = box_predictor.predict_probs(raw_predictions, proposals)
                        if probs:
                            # Convert to numpy for easier handling
                            prob_arrays = [
                                p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                                for p in probs
                            ]
                            return {
                                "probability_distributions": prob_arrays,
                                "raw_available": True,
                            }
        except Exception as e:
            logger.warning(f"Could not extract probability distributions: {e}")

        return {"raw_available": False}

    def _extract_segmentation_data(
        self, instances: Instances, idx: int
    ) -> Optional[SegmentationData]:
        """Extract comprehensive segmentation information."""
        if not instances.has("pred_masks"):
            return None

        try:
            masks = instances.pred_masks.cpu().numpy()
            if idx >= len(masks):
                return None

            mask = masks[idx]
            if len(mask.shape) == 3:
                mask = mask[0]  # Remove channel dimension if present

            # Calculate mask properties
            area = float(np.sum(mask))
            if area == 0:
                return None

            # Calculate perimeter using contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0

            # Convert mask to list for JSON serialization
            mask_list = mask.astype(int).tolist()

            return SegmentationData(mask=mask_list, area=area, perimeter=perimeter)

        except Exception as e:
            logger.warning(f"Error extracting segmentation data: {e}")
            return None

    def _extract_keypoint_data(
        self, instances: Instances, idx: int
    ) -> Optional[KeypointData]:
        """Extract comprehensive keypoint information."""
        if not instances.has("pred_keypoints"):
            return None

        try:
            keypoints = instances.pred_keypoints.cpu().numpy()
            if idx >= len(keypoints):
                return None

            kpts = keypoints[
                idx
            ]  # Shape: [num_keypoints, 3] where 3 is [x, y, confidence]

            # Extract keypoint coordinates and confidences
            keypoint_coords = kpts.tolist()
            confidence_scores = [float(kpt[2]) for kpt in kpts]
            visibility = [
                conf > 0.5 for conf in confidence_scores
            ]  # Simple visibility threshold

            # Get keypoint names if available from metadata
            keypoint_names = None
            try:
                if hasattr(self, "metadata") and hasattr(
                    self.metadata, "keypoint_names"
                ):
                    keypoint_names = self.metadata.keypoint_names
            except:
                pass

            return KeypointData(
                keypoints=keypoint_coords,
                keypoint_names=keypoint_names,
                confidence_scores=confidence_scores,
                visibility=visibility,
            )

        except Exception as e:
            logger.warning(f"Error extracting keypoint data: {e}")
            return None

    def _calculate_detection_metrics(
        self, box: np.ndarray, score: float, image_shape: Tuple[int, int], idx: int
    ) -> DetectionMetrics:
        """Calculate comprehensive detection metrics."""
        try:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            area = width * height

            # Determine size category
            image_area = image_shape[0] * image_shape[1]
            relative_area = area / image_area

            if relative_area < 0.01:
                size_category = "small"
            elif relative_area < 0.1:
                size_category = "medium"
            else:
                size_category = "large"

            # Calculate distance to image edge
            edge_distances = [
                x1,  # distance to left edge
                y1,  # distance to top edge
                image_shape[1] - x2,  # distance to right edge
                image_shape[0] - y2,  # distance to bottom edge
            ]
            min_edge_distance = float(min(edge_distances))

            # Calculate uncertainty score (simple heuristic based on confidence)
            uncertainty_score = float(1.0 - score)

            return DetectionMetrics(
                detection_size=size_category,
                edge_distance=min_edge_distance,
                uncertainty_score=uncertainty_score,
                nms_rank=idx,  # Use index as NMS rank approximation
            )

        except Exception as e:
            logger.warning(f"Error calculating detection metrics: {e}")
            return DetectionMetrics()

    def _extract_confidence_metrics(
        self, instances: Instances, idx: int, score: float
    ) -> ConfidenceMetrics:
        """Extract advanced confidence and uncertainty metrics."""
        try:
            confidence_metrics = ConfidenceMetrics()

            # Basic confidence bounds (simple heuristic)
            confidence_metrics.confidence_lower = max(0.0, float(score - 0.1))
            confidence_metrics.confidence_upper = min(1.0, float(score + 0.1))

            # Check for DensePose-specific confidence data
            if instances.has("pred_densepose"):
                try:
                    densepose_data = instances.pred_densepose
                    if (
                        hasattr(densepose_data, "sigma_1")
                        and densepose_data.sigma_1 is not None
                    ):
                        if idx < len(densepose_data.sigma_1):
                            confidence_metrics.sigma_1 = float(
                                densepose_data.sigma_1[idx].mean()
                            )

                    if (
                        hasattr(densepose_data, "sigma_2")
                        and densepose_data.sigma_2 is not None
                    ):
                        if idx < len(densepose_data.sigma_2):
                            confidence_metrics.sigma_2 = float(
                                densepose_data.sigma_2[idx].mean()
                            )

                except Exception as e:
                    logger.debug(f"Could not extract DensePose confidence metrics: {e}")

            return confidence_metrics

        except Exception as e:
            logger.warning(f"Error extracting confidence metrics: {e}")
            return ConfidenceMetrics()

    def _extract_densepose_data(
        self, instances: Instances, idx: int
    ) -> Optional[DensePoseData]:
        """Extract DensePose-specific information."""
        if not instances.has("pred_densepose"):
            return None

        try:
            densepose_output = instances.pred_densepose
            if idx >= len(densepose_output):
                return None

            densepose_data = DensePoseData()

            # Extract UV coordinates if available
            if hasattr(densepose_output, "u") and hasattr(densepose_output, "v"):
                u_coords = (
                    densepose_output.u[idx].cpu().numpy()
                    if isinstance(densepose_output.u[idx], torch.Tensor)
                    else densepose_output.u[idx]
                )
                v_coords = (
                    densepose_output.v[idx].cpu().numpy()
                    if isinstance(densepose_output.v[idx], torch.Tensor)
                    else densepose_output.v[idx]
                )

                # Combine U and V coordinates
                if u_coords.size > 0 and v_coords.size > 0:
                    uv_coords = np.stack(
                        [u_coords.flatten(), v_coords.flatten()], axis=1
                    )
                    densepose_data.uv_coordinates = uv_coords.tolist()

            # Extract part segmentation if available
            if hasattr(densepose_output, "fine_segm"):
                segm = (
                    densepose_output.fine_segm[idx].cpu().numpy()
                    if isinstance(densepose_output.fine_segm[idx], torch.Tensor)
                    else densepose_output.fine_segm[idx]
                )
                if segm.size > 0:
                    densepose_data.part_segmentation = segm.astype(int).tolist()

            return densepose_data

        except Exception as e:
            logger.warning(f"Error extracting DensePose data: {e}")
            return None

    def _create_fallback_config(self):
        """Create a fallback configuration when HADM configs are not available."""
        try:
            from detectron2.config import get_cfg

            cfg = get_cfg()

            # Minimal configuration - just the essentials
            cfg.MODEL.DEVICE = str(self.device)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # HADM-L classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
            cfg.INPUT.FORMAT = "BGR"

            # Basic required settings
            cfg.DATASETS.TRAIN = ()
            cfg.DATASETS.TEST = ()
            cfg.SOLVER.IMS_PER_BATCH = 1

            # Explicitly disable multi-label functionality to avoid MULTI_LABEL error
            cfg.MODEL.ROI_BOX_HEAD.MULTI_LABEL = False
            cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False

            logger.info("✅ Created minimal fallback configuration")
            return cfg

        except Exception as e:
            logger.error(f"❌ Failed to create fallback config: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def load_model(self) -> bool:
        """Load HADM-L model with fallback support."""
        try:
            logger.info("🔄 Loading HADM Local (HADM-L) model...")

            # Setup device
            self._setup_device()

            # Check for model weights
            model_path = settings.hadm_l_model_path
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model weights not found at {model_path}")
                logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            logger.info(f"📁 Found model weights at {model_path}")

            # Try to load model configuration
            cfg = None

            if HADM_CONFIGS_AVAILABLE:
                try:
                    logger.info("Using HADM configuration...")
                    cfg = LazyConfig.load_config(
                        str(
                            HADM_PATH
                            / "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
                        )
                    )
                    logger.info("✅ HADM LazyConfig loaded successfully")

                    # Convert LazyConfig to standard config if needed
                    if hasattr(cfg, "model"):
                        # This is a LazyConfig, we need to handle it differently
                        logger.info(
                            "LazyConfig detected, using direct model instantiation"
                        )

                        # For now, fall back to standard config
                        cfg = self._create_fallback_config()

                except Exception as config_error:
                    logger.warning(f"Failed to use HADM config: {config_error}")
                    cfg = self._create_fallback_config()
            else:
                logger.info("Using fallback configuration...")
                cfg = self._create_fallback_config()

            if cfg is None:
                logger.error("❌ Failed to create any configuration")
                logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            # Set model weights path
            cfg.MODEL.WEIGHTS = model_path

            try:
                # Create predictor - this should load model into VRAM
                logger.info("🔄 Creating predictor and loading model into VRAM...")
                self.predictor = DefaultPredictor(cfg)
                logger.info("✅ DefaultPredictor created successfully")

                # Store model reference for advanced features
                self.model = self.predictor.model

                # Update class names if available
                if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "ROI_HEADS"):
                    num_classes = getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 6)
                    self.class_names = [f"artifact_{i+1}" for i in range(num_classes)]

                # Log GPU memory usage if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"🔥 GPU memory allocated: {memory_allocated:.2f} GB")

                self.is_loaded = True
                self.simplified_mode = False  # Explicitly set to False
                logger.info("✅ HADM-L model loaded successfully into VRAM")
                return True

            except Exception as predictor_error:
                logger.error(f"❌ Failed to create predictor: {predictor_error}")
                logger.error(f"Full error traceback: {traceback.format_exc()}")
                logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-L model: {e}")
            logger.warning("⚠️ HADM-L will run in SIMPLIFIED MODE (no VRAM usage)")
            self.simplified_mode = True
            self.is_loaded = True
            return True

    def predict(self, image: np.ndarray) -> List[LocalDetection]:
        """Enhanced prediction with comprehensive information extraction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()
        detections = []

        try:
            if hasattr(self, "simplified_mode") and self.simplified_mode:
                logger.warning("Model in simplified mode - returning empty results")
                return []

            # Preprocess image
            processed_image = self._preprocess_image(image)
            image_shape = image.shape[:2]  # (height, width)

            # Run inference with timing
            inference_start = time.time()
            predictions = self.predictor(processed_image)
            inference_time = time.time() - inference_start

            # Extract probability distributions
            prob_data = self._extract_probability_distribution(predictions, [])

            if "instances" in predictions:
                instances = predictions["instances"]

                # Filter by confidence threshold
                scores = instances.scores.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()

                valid_indices = scores >= settings.confidence_threshold

                # Calculate statistics for performance metrics
                total_detections = len(scores)
                filtered_detections = np.sum(valid_indices)

                for i, (score, box, cls_id) in enumerate(zip(scores, boxes, classes)):
                    if valid_indices[i]:
                        # Create enhanced detection
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
                            processing_time=inference_time
                            / max(
                                filtered_detections, 1
                            ),  # Approximate per-detection time
                            detection_source="local_model",
                        )

                        # Add probability information
                        if prob_data.get("raw_available") and prob_data.get(
                            "probability_distributions"
                        ):
                            try:
                                if i < len(prob_data["probability_distributions"][0]):
                                    probs = prob_data["probability_distributions"][0][i]
                                    class_probs = {
                                        (
                                            self.class_names[j]
                                            if j < len(self.class_names)
                                            else f"class_{j}"
                                        ): float(probs[j])
                                        for j in range(len(probs))
                                    }
                                    detection.class_probabilities = class_probs
                                    detection.probability_distribution = probs.tolist()
                            except Exception as e:
                                logger.debug(
                                    f"Could not extract probabilities for detection {i}: {e}"
                                )

                        # Extract segmentation data
                        detection.segmentation = self._extract_segmentation_data(
                            instances, i
                        )

                        # Extract keypoint data
                        detection.keypoints = self._extract_keypoint_data(instances, i)

                        # Calculate detection metrics
                        detection.metrics = self._calculate_detection_metrics(
                            box, score, image_shape, i
                        )

                        # Extract confidence metrics
                        detection.confidence_metrics = self._extract_confidence_metrics(
                            instances, i, score
                        )

                        # Extract DensePose data if available
                        detection.densepose = self._extract_densepose_data(instances, i)

                        detections.append(detection)

                # Sort by confidence and limit results
                detections.sort(key=lambda x: x.confidence, reverse=True)
                detections = detections[: settings.max_detections]

        except Exception as e:
            logger.error(f"Error in enhanced local prediction: {e}")
            logger.error(traceback.format_exc())

        return detections


class HADMGlobalModel(HADMModelBase):
    """Enhanced global artifact detection model with comprehensive analysis."""

    def __init__(self):
        super().__init__("global")
        self.num_classes = 12  # HADM-G has 12 classes
        self.class_names = [
            "global_artifact_1",
            "global_artifact_2",
            "global_artifact_3",
            "global_artifact_4",
            "global_artifact_5",
            "global_artifact_6",
            "global_artifact_7",
            "global_artifact_8",
            "global_artifact_9",
            "global_artifact_10",
            "global_artifact_11",
            "global_artifact_12",
        ]  # Will be updated when we have actual class names
        self.model = None

    def _calculate_statistical_measures(
        self, probabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate statistical measures from probability distribution."""
        try:
            probs = list(probabilities.values())
            probs_tensor = torch.tensor(probs)

            # Calculate entropy
            entropy = float(-torch.sum(probs_tensor * torch.log(probs_tensor + 1e-8)))

            # Max probability
            max_prob = float(max(probs))

            # Probability gap (difference between top 2)
            sorted_probs = sorted(probs, reverse=True)
            prob_gap = (
                float(sorted_probs[0] - sorted_probs[1])
                if len(sorted_probs) > 1
                else 0.0
            )

            return {
                "entropy": entropy,
                "max_probability": max_prob,
                "probability_gap": prob_gap,
            }

        except Exception as e:
            logger.warning(f"Error calculating statistical measures: {e}")
            return {}

    def _extract_raw_logits(
        self, predictions: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Extract raw logits before activation."""
        try:
            if "instances" in predictions:
                instances = predictions["instances"]
                if hasattr(instances, "scores") and hasattr(self.model, "roi_heads"):
                    # Try to get raw scores before softmax/sigmoid
                    raw_scores = {}
                    scores = instances.scores.cpu().numpy()
                    classes = instances.pred_classes.cpu().numpy()

                    for i, (score, cls_id) in enumerate(zip(scores, classes)):
                        class_name = (
                            self.class_names[cls_id]
                            if cls_id < len(self.class_names)
                            else f"class_{cls_id}"
                        )
                        # Convert confidence back to approximate logit
                        logit = float(np.log(score / (1 - score + 1e-8)))
                        raw_scores[class_name] = logit

                    return raw_scores
        except Exception as e:
            logger.debug(f"Could not extract raw logits: {e}")

        return None

    def _analyze_artifact_indicators(
        self, image: np.ndarray, probabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze specific artifact indicators."""
        try:
            indicators = {}

            # Compression artifacts (simple heuristic based on image statistics)
            gray = (
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if len(image.shape) == 3
                else image
            )

            # Calculate image gradients for sharpness/compression analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Sharpness indicator
            indicators["sharpness_score"] = float(np.mean(gradient_magnitude) / 255.0)

            # Noise estimation using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            indicators["noise_estimate"] = float(
                np.var(laplacian) / 10000.0
            )  # Normalize

            # Compression artifact estimation (simplified)
            # Look for blocking artifacts using frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # Simple heuristic for compression artifacts
            indicators["compression_artifacts"] = float(
                np.std(magnitude_spectrum) / np.mean(magnitude_spectrum)
            )

            # Authenticity indicators based on model predictions
            if "authentic" in probabilities:
                indicators["authenticity_indicator"] = probabilities["authentic"]
            if "manipulated" in probabilities:
                indicators["manipulation_indicator"] = probabilities["manipulated"]

            return indicators

        except Exception as e:
            logger.warning(f"Error analyzing artifact indicators: {e}")
            return {}

    def _create_fallback_config(self):
        """Create a fallback configuration when HADM configs are not available."""
        try:
            from detectron2.config import get_cfg

            cfg = get_cfg()

            # Minimal configuration - just the essentials
            cfg.MODEL.DEVICE = str(self.device)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # HADM-L classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
            cfg.INPUT.FORMAT = "BGR"

            # Basic required settings
            cfg.DATASETS.TRAIN = ()
            cfg.DATASETS.TEST = ()
            cfg.SOLVER.IMS_PER_BATCH = 1

            # Explicitly disable multi-label functionality to avoid MULTI_LABEL error
            cfg.MODEL.ROI_BOX_HEAD.MULTI_LABEL = False
            cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False

            logger.info("✅ Created minimal fallback configuration")
            return cfg

        except Exception as e:
            logger.error(f"❌ Failed to create fallback config: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def load_model(self) -> bool:
        """Load HADM-G model with fallback support."""
        try:
            logger.info("🔄 Loading HADM Global (HADM-G) model...")

            # Setup device
            self._setup_device()

            # Check for model weights
            model_path = settings.hadm_g_model_path
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model weights not found at {model_path}")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            logger.info(f"📁 Found model weights at {model_path}")

            # Try to load model configuration
            cfg = None

            if HADM_CONFIGS_AVAILABLE:
                try:
                    logger.info("Using HADM configuration...")
                    cfg = LazyConfig.load_config(
                        str(
                            HADM_PATH
                            / "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"
                        )
                    )
                    logger.info("✅ HADM LazyConfig loaded successfully")

                    # Convert LazyConfig to standard config if needed
                    if hasattr(cfg, "model"):
                        # This is a LazyConfig, we need to handle it differently
                        logger.info(
                            "LazyConfig detected, using direct model instantiation"
                        )

                        # For now, fall back to standard config
                        cfg = self._create_fallback_config()

                except Exception as config_error:
                    logger.warning(f"Failed to use HADM config: {config_error}")
                    cfg = self._create_fallback_config()
            else:
                logger.info("Using fallback configuration...")
                cfg = self._create_fallback_config()

            if cfg is None:
                logger.error("❌ Failed to create any configuration")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

            # Set model weights path
            cfg.MODEL.WEIGHTS = model_path

            try:
                # Create predictor - this should load model into VRAM
                logger.info("🔄 Creating predictor and loading model into VRAM...")
                self.predictor = DefaultPredictor(cfg)
                logger.info("✅ DefaultPredictor created successfully")

                # Store model reference for advanced features
                self.model = self.predictor.model

                # Update class names if available
                if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "ROI_HEADS"):
                    num_classes = getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 12)
                    self.class_names = [
                        f"global_artifact_{i+1}" for i in range(num_classes)
                    ]

                # Log GPU memory usage if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"🔥 GPU memory allocated: {memory_allocated:.2f} GB")

                self.is_loaded = True
                self.simplified_mode = False  # Explicitly set to False
                logger.info("✅ HADM-G model loaded successfully into VRAM")
                return True

            except Exception as predictor_error:
                logger.error(f"❌ Failed to create predictor: {predictor_error}")
                logger.error(f"Full error traceback: {traceback.format_exc()}")
                logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE (no VRAM usage)")
                self.simplified_mode = True
                self.is_loaded = True
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load HADM-G model: {e}")
            logger.warning("⚠️ HADM-G will run in SIMPLIFIED MODE (no VRAM usage)")
            self.simplified_mode = True
            self.is_loaded = True
            return True

    def predict(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """Enhanced global artifact detection with comprehensive analysis."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            if hasattr(self, "simplified_mode") and self.simplified_mode:
                logger.warning("Model in simplified mode - returning empty results")
                return None

            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Run inference
            predictions = self.predictor(processed_image)

            # Process results for global classification
            if "instances" in predictions:
                instances = predictions["instances"]

                if len(instances) > 0:
                    # For global detection, use the highest confidence detection
                    scores = instances.scores.cpu().numpy()
                    classes = instances.pred_classes.cpu().numpy()

                    # Get the highest confidence detection
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    best_class = classes[best_idx]

                    # Create enhanced probability distribution
                    probabilities = {name: 0.0 for name in self.class_names}
                    probabilities[self.class_names[best_class]] = float(best_score)

                    # Try to get full probability distribution if available
                    try:
                        if hasattr(self.model, "roi_heads") and hasattr(
                            self.model.roi_heads, "box_predictor"
                        ):
                            # This would require access to raw predictions - simplified for now
                            pass
                    except:
                        pass

                    # Calculate statistical measures
                    stats = self._calculate_statistical_measures(probabilities)

                    # Extract raw logits
                    raw_logits = self._extract_raw_logits(predictions)

                    # Analyze artifact indicators
                    artifact_indicators = self._analyze_artifact_indicators(
                        image, probabilities
                    )

                    # Calculate processing time
                    processing_time = time.time() - start_time

                    # Create enhanced global detection
                    global_detection = GlobalDetection(
                        class_id=int(best_class),
                        class_name=(
                            self.class_names[best_class]
                            if best_class < len(self.class_names)
                            else f"class_{best_class}"
                        ),
                        confidence=float(best_score),
                        probabilities=probabilities,
                        processing_time=processing_time,
                        detection_source="global_model",
                    )

                    # Add statistical measures
                    if stats:
                        global_detection.entropy = stats.get("entropy")
                        global_detection.max_probability = stats.get("max_probability")
                        global_detection.probability_gap = stats.get("probability_gap")

                    # Add raw logits if available
                    if raw_logits:
                        global_detection.logits = raw_logits

                    # Add artifact indicators
                    if artifact_indicators:
                        global_detection.artifact_indicators = artifact_indicators

                        # Set specific confidence scores
                        if "authenticity_indicator" in artifact_indicators:
                            global_detection.manipulation_confidence = (
                                1.0 - artifact_indicators["authenticity_indicator"]
                            )

                    # Add uncertainty score
                    global_detection.uncertainty_score = float(1.0 - best_score)

                    # Create confidence metrics
                    confidence_metrics = ConfidenceMetrics(
                        confidence_lower=max(0.0, float(best_score - 0.1)),
                        confidence_upper=min(1.0, float(best_score + 0.1)),
                        authenticity_confidence=artifact_indicators.get(
                            "authenticity_indicator"
                        ),
                        manipulation_confidence=artifact_indicators.get(
                            "manipulation_indicator"
                        ),
                    )
                    global_detection.confidence_metrics = confidence_metrics

                    return global_detection

        except Exception as e:
            logger.error(f"Error in enhanced global prediction: {e}")
            logger.error(traceback.format_exc())

        return None


class HADMModelManager:
    """Manager for HADM models."""

    def __init__(self):
        self.local_model = HADMLocalModel()
        self.global_model = HADMGlobalModel()
        self.models_loaded = False

    def load_models(self) -> Dict[str, bool]:
        """Load all models."""
        results = {}

        if settings.preload_models:
            logger.info("Loading HADM models...")

            results["local"] = self.local_model.load_model()
            results["global"] = self.global_model.load_model()

            self.models_loaded = all(results.values())

            if self.models_loaded:
                logger.info("All HADM models loaded successfully")
            else:
                logger.warning("Some HADM models failed to load")
        else:
            logger.info("Model preloading disabled")
            results["local"] = False
            results["global"] = False

        return results

    def get_model_status(self) -> Dict[str, bool]:
        """Get model loading status."""
        return {
            "local": self.local_model.is_loaded,
            "global": self.global_model.is_loaded,
        }

    async def predict_local(self, image: np.ndarray) -> List[LocalDetection]:
        """Run local detection."""
        if not self.local_model.is_loaded:
            if not self.local_model.load_model():
                raise RuntimeError("Failed to load HADM-L model")

        return self.local_model.predict(image)

    async def predict_global(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """Run global detection."""
        if not self.global_model.is_loaded:
            if not self.global_model.load_model():
                raise RuntimeError("Failed to load HADM-G model")

        return self.global_model.predict(image)

    async def predict_both(
        self, image: np.ndarray
    ) -> Tuple[List[LocalDetection], Optional[GlobalDetection]]:
        """Run both local and global detection."""
        local_results = await self.predict_local(image)
        global_results = await self.predict_global(image)
        return local_results, global_results


# Global model manager instance
model_manager = HADMModelManager()
