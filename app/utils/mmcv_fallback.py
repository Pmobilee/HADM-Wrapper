"""
MMCV Fallback Module
Provides fallback implementations for mmcv functionality when mmcv is not available or incomplete.
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def soft_nms(boxes, scores, iou_threshold=0.3, sigma=0.5, min_score=1e-3, method='linear'):
    """
    Fallback soft NMS implementation using regular NMS.
    
    Args:
        boxes: Tensor of shape (N, 4) containing bounding boxes
        scores: Tensor of shape (N,) containing scores
        iou_threshold: IoU threshold for NMS
        sigma: Sigma parameter for soft NMS (not used in fallback)
        min_score: Minimum score threshold
        method: Method for soft NMS (not used in fallback)
    
    Returns:
        dets: Tensor of shape (M, 5) containing [x1, y1, x2, y2, score]
        keep: Tensor of shape (M,) containing indices of kept boxes
    """
    logger.warning("Using fallback NMS instead of soft NMS (mmcv not available)")
    
    from torchvision.ops import nms
    
    # Apply score threshold
    valid_mask = scores > min_score
    valid_boxes = boxes[valid_mask]
    valid_scores = scores[valid_mask]
    valid_indices = torch.nonzero(valid_mask).flatten()
    
    if len(valid_boxes) == 0:
        # Return empty results
        empty_dets = torch.zeros((0, 5), dtype=boxes.dtype, device=boxes.device)
        empty_keep = torch.zeros((0,), dtype=torch.long, device=boxes.device)
        return empty_dets, empty_keep
    
    # Apply regular NMS
    keep_indices = nms(valid_boxes, valid_scores, iou_threshold)
    
    # Get kept boxes and scores
    kept_boxes = valid_boxes[keep_indices]
    kept_scores = valid_scores[keep_indices]
    
    # Combine boxes and scores like soft_nms output
    dets = torch.cat([kept_boxes, kept_scores.unsqueeze(1)], dim=1)
    
    # Map back to original indices
    original_keep = valid_indices[keep_indices]
    
    return dets, original_keep

def setup_mmcv_fallback():
    """Setup fallback mmcv modules in sys.modules to prevent import errors."""
    import sys
    import types
    
    try:
        # Try to import mmcv first
        import mmcv
        from mmcv import ops
        logger.info("✅ MMCV and ops available - no fallback needed")
        return True
    except ImportError:
        logger.warning("⚠️ MMCV not available - setting up fallback modules")
        
        # Create mock mmcv module structure
        mock_mmcv = types.ModuleType('mmcv')
        mock_ops = types.ModuleType('mmcv.ops')
        
        # Add fallback functions
        mock_ops.soft_nms = soft_nms
        mock_mmcv.ops = mock_ops
        mock_mmcv.__version__ = "fallback-1.0.0"
        
        # Register in sys.modules
        sys.modules['mmcv'] = mock_mmcv
        sys.modules['mmcv.ops'] = mock_ops
        
        logger.info("✅ MMCV fallback modules registered")
        return False

# Auto-setup when module is imported
setup_mmcv_fallback() 