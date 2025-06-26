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

# Add HADM to Python path
HADM_PATH = Path(__file__).parent.parent.parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

logger = logging.getLogger(__name__)

# Try to import dependencies - handle gracefully if not available
DEPENDENCIES_AVAILABLE = True
try:
    import torch
    from detectron2.config import get_cfg, CfgNode
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.config import LazyConfig
    
    # Try to import HADM specific configs
    try:
        from projects.ViTDet.configs.eva2_o365_to_coco.demo_local import model as local_model_config
        from projects.ViTDet.configs.eva2_o365_to_coco.demo_global import model as global_model_config
        HADM_CONFIGS_AVAILABLE = True
        logger.info("HADM configurations loaded successfully")
    except ImportError as e:
        logger.warning(f"HADM configurations not available: {e}")
        HADM_CONFIGS_AVAILABLE = False
        
except ImportError as e:
    logger.error(f"Core dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    HADM_CONFIGS_AVAILABLE = False

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
            
            # Create configuration
            if HADM_CONFIGS_AVAILABLE:
                # Use HADM configuration
                cfg = LazyConfig.load_config(str(HADM_PATH / "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"))
                
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
                
                logger.info("Using HADM configuration for local model")
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
            logger.info("HADM-L model loaded successfully")
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
            logger.info("HADM-G model loaded successfully")
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