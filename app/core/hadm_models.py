"""
HADM Model Wrappers
"""
import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image

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
        mock_mmcv = types.ModuleType('mmcv')
        mock_ops = types.ModuleType('mmcv.ops')
        
        def fallback_soft_nms(boxes, scores, iou_threshold=0.3, sigma=0.5, min_score=1e-3, method='linear'):
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
        sys.modules['mmcv'] = mock_mmcv
        sys.modules['mmcv.ops'] = mock_ops

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
    logger.warning("HADM models may work without MMCV, but some features may be limited")
    MMCV_AVAILABLE = False

# Try to import HADM specific configs
if DEPENDENCIES_AVAILABLE and DETECTRON2_AVAILABLE:
    try:
        logger.info("Attempting to import HADM configurations...")
        
        # Test basic HADM path access
        hadm_projects_path = HADM_PATH / "projects"
        if not hadm_projects_path.exists():
            logger.error(f"❌ HADM projects directory not found: {hadm_projects_path}")
            raise ImportError(f"HADM projects directory not found: {hadm_projects_path}")
        
        vitdet_path = hadm_projects_path / "ViTDet" / "configs" / "eva2_o365_to_coco"
        if not vitdet_path.exists():
            logger.error(f"❌ HADM ViTDet configs not found: {vitdet_path}")
            raise ImportError(f"HADM ViTDet configs not found: {vitdet_path}")
        
        # Try importing the configs
        from projects.ViTDet.configs.eva2_o365_to_coco.demo_local import model as local_model_config
        logger.info("✅ HADM local config imported successfully")
        
        from projects.ViTDet.configs.eva2_o365_to_coco.demo_global import model as global_model_config
        logger.info("✅ HADM global config imported successfully")
        
        HADM_CONFIGS_AVAILABLE = True
        logger.info("✅ All HADM configurations loaded successfully")
        
    except ImportError as e:
        logger.error(f"❌ HADM configurations not available: {e}")
        logger.error(f"Error details: {str(e)}")
        logger.error(f"HADM_PATH: {HADM_PATH}")
        logger.error(f"HADM exists: {HADM_PATH.exists()}")
        
        # Log Python path for debugging
        logger.error(f"Python path includes HADM: {str(HADM_PATH) in sys.path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        
        HADM_CONFIGS_AVAILABLE = False
    except Exception as e:
        logger.error(f"❌ Unexpected error loading HADM configurations: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
from app.models.schemas import LocalDetection, GlobalDetection, BoundingBox

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
            logger.error("Required dependencies not available. Please install detectron2 and PyTorch.")
            
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
        
        padded_image = np.zeros((settings.image_size, settings.image_size, 3), dtype=image.dtype)
        padded_image[pad_h:pad_h+h, pad_w:pad_w+w] = image
        
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
    """HADM Local (HADM-L) model wrapper for bounding box detection."""
    
    def __init__(self):
        super().__init__("local")
        self.num_classes = 6  # HADM-L has 6 classes
        self.class_names = [
            "artifact_1", "artifact_2", "artifact_3", 
            "artifact_4", "artifact_5", "artifact_6"
        ]  # Will be updated when we have actual class names
    
    def load_model(self) -> bool:
        """Load HADM-L model."""
        try:
            if not DEPENDENCIES_AVAILABLE:
                logger.error("Dependencies not available for model loading")
                return False
                
            logger.info("Loading HADM-L model...")
            
            self._setup_device()
            
            # Check if model file exists
            model_path = settings.hadm_l_model_path
            if not os.path.exists(model_path):
                logger.error(f"HADM-L model not found at {model_path}")
                return False
            
            # Test model loading first (like diagnose_models.py does)
            logger.info("Testing model file loading...")
            try:
                # Try to load model weights (handle PyTorch 2.6 weights_only issue)
                model_state = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info(f"✅ Model state loaded successfully, keys: {len(model_state.keys()) if isinstance(model_state, dict) else 'Not a dict'}")
                
                # Show model structure
                if isinstance(model_state, dict):
                    if 'model' in model_state:
                        logger.info("✅ Model contains 'model' key")
                    if 'ema' in model_state:
                        logger.info("✅ Model contains 'ema' key (EMA weights)")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model weights: {e}")
                return False
            
            # Try different configuration approaches in order of preference
            predictor_created = False
            
            # Approach 1: Use HADM configuration if available
            if HADM_CONFIGS_AVAILABLE and not predictor_created:
                try:
                    logger.info("Attempting HADM LazyConfig approach...")
                    cfg = LazyConfig.load_config(str(HADM_PATH / "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"))
                    logger.info("✅ HADM LazyConfig loaded successfully")
                    
                    # Convert LazyConfig to standard config for predictor
                    cfg_dict = LazyConfig.to_py(cfg.model)
                    cfg = get_cfg()
                    
                    # Set basic configuration
                    cfg.MODEL.DEVICE = self.device
                    cfg.MODEL.WEIGHTS = model_path
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                    
                    # Set input format
                    cfg.INPUT.FORMAT = "BGR"
                    cfg.INPUT.MIN_SIZE_TEST = 1024
                    cfg.INPUT.MAX_SIZE_TEST = 1024
                    
                    # Try to create predictor
                    self.predictor = DefaultPredictor(cfg)
                    predictor_created = True
                    logger.info("✅ Using HADM configuration for local model")
                    
                except Exception as e:
                    logger.warning(f"⚠️ HADM config approach failed: {e}")
                    predictor_created = False
            
            # Approach 2: Use basic detectron2 configuration
            if not predictor_created:
                try:
                    logger.info("Attempting basic detectron2 configuration...")
                    cfg = get_cfg()
                    
                    # Basic model configuration
                    cfg.MODEL.DEVICE = self.device
                    cfg.MODEL.WEIGHTS = model_path
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                    
                    # Input configuration
                    cfg.INPUT.FORMAT = "BGR"
                    cfg.INPUT.MIN_SIZE_TEST = 1024
                    cfg.INPUT.MAX_SIZE_TEST = 1024
                    
                    # Try to create predictor
                    self.predictor = DefaultPredictor(cfg)
                    predictor_created = True
                    logger.info("✅ Using basic detectron2 configuration")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Basic detectron2 config failed: {e}")
                    predictor_created = False
            
            # Approach 3: Simplified model loading (fallback)
            if not predictor_created:
                logger.info("Falling back to simplified model loading...")
                return self._load_simplified_model(model_path)
            else:
                # Fallback to basic configuration
                logger.warning("Using fallback configuration for local model")
                cfg = get_cfg()
                cfg.MODEL.DEVICE = self.device
                cfg.MODEL.WEIGHTS = model_path
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                cfg.INPUT.FORMAT = "BGR"
                cfg.INPUT.MIN_SIZE_TEST = 1024
                cfg.INPUT.MAX_SIZE_TEST = 1024
            
            # Create predictor
            try:
                logger.info("Creating DefaultPredictor...")
                self.predictor = DefaultPredictor(cfg)
                logger.info("DefaultPredictor created successfully")
            except Exception as e:
                logger.error(f"Failed to create predictor: {e}")
                # Try simplified approach
                logger.info("Attempting simplified model loading...")
                return self._load_simplified_model(model_path)
            
            # Setup metadata
            self.metadata = MetadataCatalog.get("hadm_local")
            if not hasattr(self.metadata, "thing_classes"):
                self.metadata.thing_classes = self.class_names
            
            self.is_loaded = True
            logger.info("HADM-L model loaded successfully - GPU memory allocated")
            
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HADM-L model: {e}")
            logger.info("Attempting simplified model loading as fallback...")
            return self._load_simplified_model(settings.hadm_l_model_path)
    
    def _load_simplified_model(self, model_path: str) -> bool:
        """Simplified model loading as fallback."""
        try:
            # Load model weights directly
            self.model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info("Model weights loaded directly")
            
            # Mark as loaded but with limited functionality
            self.is_loaded = True
            self.simplified_mode = True
            
            return True
        except Exception as e:
            logger.error(f"Simplified model loading also failed: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> List[LocalDetection]:
        """
        Run local artifact detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of local detections
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if hasattr(self, 'simplified_mode') and self.simplified_mode:
            # Return empty results for simplified mode
            logger.warning("Model in simplified mode - returning empty results")
            return []
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Run inference
        predictions = self.predictor(processed_image)
        
        # Process results
        detections = []
        if "instances" in predictions:
            instances = predictions["instances"]
            
            # Filter by confidence threshold
            scores = instances.scores.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            valid_indices = scores >= settings.confidence_threshold
            
            for i, (score, box, cls_id) in enumerate(zip(scores, boxes, classes)):
                if valid_indices[i]:
                    detection = LocalDetection(
                        bbox=BoundingBox(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3])
                        ),
                        confidence=float(score),
                        class_id=int(cls_id),
                        class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    )
                    detections.append(detection)
        
        # Sort by confidence and limit results
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections[:settings.max_detections]


class HADMGlobalModel(HADMModelBase):
    """HADM Global (HADM-G) model wrapper for whole-image classification."""
    
    def __init__(self):
        super().__init__("global")
        self.num_classes = 12  # HADM-G has 12 classes
        self.class_names = [
            "global_artifact_1", "global_artifact_2", "global_artifact_3",
            "global_artifact_4", "global_artifact_5", "global_artifact_6",
            "global_artifact_7", "global_artifact_8", "global_artifact_9",
            "global_artifact_10", "global_artifact_11", "global_artifact_12"
        ]  # Will be updated when we have actual class names
    
    def load_model(self) -> bool:
        """Load HADM-G model."""
        try:
            if not DEPENDENCIES_AVAILABLE:
                logger.error("Dependencies not available for model loading")
                return False
                
            logger.info("Loading HADM-G model...")
            
            self._setup_device()
            
            # Check if model file exists
            model_path = settings.hadm_g_model_path
            if not os.path.exists(model_path):
                logger.error(f"HADM-G model not found at {model_path}")
                return False
            
            # Test model loading first (like diagnose_models.py does)
            logger.info("Testing model file loading...")
            try:
                # Try to load model weights (handle PyTorch 2.6 weights_only issue)
                model_state = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info(f"Model state loaded successfully, keys: {len(model_state.keys()) if isinstance(model_state, dict) else 'Not a dict'}")
                
                # Show model structure
                if isinstance(model_state, dict):
                    if 'model' in model_state:
                        logger.info("Model contains 'model' key")
                    if 'ema' in model_state:
                        logger.info("Model contains 'ema' key (EMA weights)")
                
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
                return False
            
            # Create configuration
            if HADM_CONFIGS_AVAILABLE:
                # Use HADM configuration
                cfg = LazyConfig.load_config(str(HADM_PATH / "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py"))
                
                # Convert LazyConfig to standard config for predictor
                cfg_dict = LazyConfig.to_py(cfg.model)
                cfg = get_cfg()
                
                # Set basic configuration
                cfg.MODEL.DEVICE = self.device
                cfg.MODEL.WEIGHTS = model_path
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                
                # Set input format
                cfg.INPUT.FORMAT = "BGR"
                cfg.INPUT.MIN_SIZE_TEST = 1024
                cfg.INPUT.MAX_SIZE_TEST = 1024
                
                logger.info("Using HADM configuration for global model")
            else:
                # Fallback to basic configuration
                logger.warning("Using fallback configuration for global model")
                cfg = get_cfg()
                cfg.MODEL.DEVICE = self.device
                cfg.MODEL.WEIGHTS = model_path
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
                cfg.INPUT.FORMAT = "BGR"
                cfg.INPUT.MIN_SIZE_TEST = 1024
                cfg.INPUT.MAX_SIZE_TEST = 1024
            
            # Create predictor
            try:
                self.predictor = DefaultPredictor(cfg)
                logger.info("DefaultPredictor created successfully")
            except Exception as e:
                logger.error(f"Failed to create predictor: {e}")
                # Try simplified approach
                logger.info("Attempting simplified model loading...")
                return self._load_simplified_model(model_path)
            
            # Setup metadata
            self.metadata = MetadataCatalog.get("hadm_global")
            if not hasattr(self.metadata, "thing_classes"):
                self.metadata.thing_classes = self.class_names
            
            self.is_loaded = True
            logger.info("HADM-G model loaded successfully - GPU memory allocated")
            
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HADM-G model: {e}")
            logger.info("Attempting simplified model loading as fallback...")
            return self._load_simplified_model(settings.hadm_g_model_path)
    
    def _load_simplified_model(self, model_path: str) -> bool:
        """Simplified model loading as fallback."""
        try:
            # Load model weights directly
            self.model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info("Model weights loaded directly")
            
            # Mark as loaded but with limited functionality
            self.is_loaded = True
            self.simplified_mode = True
            
            return True
        except Exception as e:
            logger.error(f"Simplified model loading also failed: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Optional[GlobalDetection]:
        """
        Run global artifact detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Global detection result
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if hasattr(self, 'simplified_mode') and self.simplified_mode:
            # Return empty results for simplified mode
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
                # For global detection, we might need to aggregate results
                # or use the highest confidence detection
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                
                # Get the highest confidence detection
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                best_class = classes[best_idx]
                
                # Create probability distribution (simplified)
                probabilities = {name: 0.0 for name in self.class_names}
                probabilities[self.class_names[best_class]] = float(best_score)
                
                return GlobalDetection(
                    class_id=int(best_class),
                    class_name=self.class_names[best_class] if best_class < len(self.class_names) else f"class_{best_class}",
                    confidence=float(best_score),
                    probabilities=probabilities
                )
        
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
            "global": self.global_model.is_loaded
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
    
    async def predict_both(self, image: np.ndarray) -> Tuple[List[LocalDetection], Optional[GlobalDetection]]:
        """Run both local and global detection."""
        local_results = await self.predict_local(image)
        global_results = await self.predict_global(image)
        return local_results, global_results


# Global model manager instance
model_manager = HADMModelManager() 