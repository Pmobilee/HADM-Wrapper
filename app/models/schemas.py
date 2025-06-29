"""
HADM Server API Schemas
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class DetectionType(str, Enum):
    """Detection type enumeration."""

    LOCAL = "local"
    GLOBAL = "global"
    BOTH = "both"


class BoundingBox(BaseModel):
    """Enhanced bounding box coordinates with additional metrics."""

    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

    # Additional computed properties
    center_x: Optional[float] = Field(None, description="Center X coordinate")
    center_y: Optional[float] = Field(None, description="Center Y coordinate")
    width: Optional[float] = Field(None, description="Bounding box width")
    height: Optional[float] = Field(None, description="Bounding box height")
    area: Optional[float] = Field(None, description="Bounding box area")
    aspect_ratio: Optional[float] = Field(None, description="Width/height ratio")

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-compute derived properties
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0.0


class SegmentationMask(BaseModel):
    """Segmentation mask information."""

    rle_encoding: Optional[str] = Field(None, description="RLE encoded mask")
    polygon: Optional[List[List[float]]] = Field(
        None, description="Polygon coordinates"
    )
    area: Optional[float] = Field(None, description="Mask area in pixels")
    bbox_coverage: Optional[float] = Field(
        None, description="Percentage of bbox covered by mask"
    )


class KeypointData(BaseModel):
    """Keypoint detection information."""

    keypoints: List[List[float]] = Field(
        ..., description="Keypoint coordinates [x, y, confidence]"
    )
    keypoint_names: Optional[List[str]] = Field(
        None, description="Names of detected keypoints"
    )
    confidence_scores: Optional[List[float]] = Field(
        None, description="Per-keypoint confidence"
    )
    heatmaps: Optional[List[List[List[float]]]] = Field(
        None, description="Raw keypoint heatmaps"
    )
    visibility: Optional[List[bool]] = Field(
        None, description="Keypoint visibility flags"
    )


class DetectionMetrics(BaseModel):
    """Additional detection metrics and statistics."""

    objectness_score: Optional[float] = Field(None, description="Objectness confidence")
    iou_with_gt: Optional[float] = Field(
        None, description="IoU with ground truth (if available)"
    )
    nms_rank: Optional[int] = Field(None, description="Rank after NMS")
    detection_size: Optional[str] = Field(
        None, description="Size category (small/medium/large)"
    )
    edge_distance: Optional[float] = Field(None, description="Distance to image edge")
    overlap_ratio: Optional[float] = Field(
        None, description="Overlap with other detections"
    )
    # New advanced metrics
    raw_logits: Optional[Dict[str, float]] = Field(
        None, description="Raw logits before activation"
    )
    uncertainty_score: Optional[float] = Field(
        None, description="Model uncertainty estimate"
    )
    feature_vector: Optional[List[float]] = Field(
        None, description="Raw feature representation"
    )
    anchor_info: Optional[Dict[str, Any]] = Field(
        None, description="Anchor box information"
    )
    multi_scale_scores: Optional[Dict[str, float]] = Field(
        None, description="Scores at different scales"
    )


class SegmentationData(BaseModel):
    """Segmentation mask information."""

    mask: Optional[List[List[int]]] = Field(None, description="Binary mask as 2D array")
    mask_rle: Optional[str] = Field(None, description="Run-length encoded mask")
    mask_confidence: Optional[List[List[float]]] = Field(
        None, description="Per-pixel confidence"
    )
    coarse_mask: Optional[List[List[int]]] = Field(
        None, description="Lower resolution mask"
    )
    fine_mask: Optional[List[List[int]]] = Field(
        None, description="Higher resolution mask"
    )
    area: Optional[float] = Field(None, description="Mask area in pixels")
    perimeter: Optional[float] = Field(None, description="Mask perimeter")


class ConfidenceMetrics(BaseModel):
    """Advanced confidence and uncertainty metrics."""

    # Statistical confidence measures
    sigma_1: Optional[float] = Field(None, description="Primary uncertainty estimate")
    sigma_2: Optional[float] = Field(None, description="Secondary uncertainty estimate")
    kappa_u: Optional[float] = Field(None, description="U-direction confidence")
    kappa_v: Optional[float] = Field(None, description="V-direction confidence")

    # Confidence intervals
    confidence_lower: Optional[float] = Field(
        None, description="Lower confidence bound"
    )
    confidence_upper: Optional[float] = Field(
        None, description="Upper confidence bound"
    )

    # Segmentation confidences
    coarse_segm_confidence: Optional[float] = Field(
        None, description="Coarse segmentation confidence"
    )
    fine_segm_confidence: Optional[float] = Field(
        None, description="Fine segmentation confidence"
    )

    # Model-specific confidences
    authenticity_confidence: Optional[float] = Field(
        None, description="Authenticity assessment"
    )
    manipulation_confidence: Optional[float] = Field(
        None, description="Manipulation detection confidence"
    )


class DensePoseData(BaseModel):
    """DensePose-specific information."""

    uv_coordinates: Optional[List[List[float]]] = Field(
        None, description="UV texture coordinates"
    )
    part_segmentation: Optional[List[List[int]]] = Field(
        None, description="Body part segmentation"
    )
    dense_correspondences: Optional[Dict[str, Any]] = Field(
        None, description="Dense correspondence data"
    )
    surface_normals: Optional[List[List[float]]] = Field(
        None, description="Surface normal vectors"
    )


class AttentionData(BaseModel):
    """Attention and feature visualization data."""

    attention_maps: Optional[List[List[List[float]]]] = Field(
        None, description="Model attention maps"
    )
    feature_maps: Optional[Dict[str, List[List[List[float]]]]] = Field(
        None, description="Feature maps by layer"
    )
    gradient_maps: Optional[List[List[List[float]]]] = Field(
        None, description="Gradient-based attention"
    )
    activation_maps: Optional[Dict[str, Any]] = Field(
        None, description="Layer activation maps"
    )


class LocalDetection(BaseModel):
    """Enhanced local artifact detection result."""

    # Core detection info
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    class_id: int = Field(..., description="Detected class ID")
    class_name: str = Field(..., description="Detected class name")

    # Enhanced probability information
    class_probabilities: Optional[Dict[str, float]] = Field(
        None, description="All class probabilities"
    )
    raw_scores: Optional[Dict[str, float]] = Field(
        None, description="Raw prediction scores"
    )
    probability_distribution: Optional[List[float]] = Field(
        None, description="Full probability vector"
    )

    # Segmentation information
    segmentation: Optional[SegmentationData] = Field(
        None, description="Segmentation data"
    )

    # Keypoint information
    keypoints: Optional[KeypointData] = Field(
        None, description="Keypoint data if available"
    )

    # Advanced confidence metrics
    confidence_metrics: Optional[ConfidenceMetrics] = Field(
        None, description="Advanced confidence measures"
    )

    # Detection metadata
    metrics: Optional[DetectionMetrics] = Field(
        None, description="Additional detection metrics"
    )

    # Specialized data
    densepose: Optional[DensePoseData] = Field(
        None, description="DensePose data if available"
    )
    attention: Optional[AttentionData] = Field(
        None, description="Attention and visualization data"
    )

    # Model-specific scores
    artifact_severity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Artifact severity score"
    )
    authenticity_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Authenticity confidence"
    )

    # Processing metadata
    processing_time: Optional[float] = Field(
        None, description="Time to process this detection"
    )
    detection_features: Optional[Dict[str, float]] = Field(
        None, description="Raw detection features"
    )
    detection_source: Optional[str] = Field(
        None, description="Which model/stage produced this detection"
    )


class GlobalClassification(BaseModel):
    """Enhanced global classification with detailed analysis."""

    class_id: int = Field(..., description="Predicted class ID")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )

    # Detailed probability distribution
    all_class_probabilities: Dict[str, float] = Field(
        ..., description="All class probabilities"
    )
    top_k_predictions: List[Dict[str, Union[str, float]]] = Field(
        ..., description="Top K predictions with scores"
    )

    # Confidence analysis
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="Confidence interval bounds"
    )
    entropy: Optional[float] = Field(
        None, description="Prediction entropy (uncertainty measure)"
    )
    max_probability_gap: Optional[float] = Field(
        None, description="Gap between top 2 predictions"
    )

    # Feature analysis
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )
    attention_weights: Optional[List[List[float]]] = Field(
        None, description="Attention map weights"
    )

    # HADM-specific scores
    manipulation_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Manipulation detection confidence"
    )
    authenticity_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Overall authenticity score"
    )
    artifact_types: Optional[List[str]] = Field(
        None, description="Types of artifacts detected"
    )


class GlobalDetection(BaseModel):
    """Enhanced global artifact detection result."""

    # Core classification info
    class_id: int = Field(..., description="Detected class ID")
    class_name: str = Field(..., description="Detected class name")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )

    # Enhanced probability information
    probabilities: Dict[str, float] = Field(..., description="All class probabilities")
    raw_scores: Optional[Dict[str, float]] = Field(
        None, description="Raw prediction scores"
    )
    probability_distribution: Optional[List[float]] = Field(
        None, description="Full probability vector"
    )
    logits: Optional[Dict[str, float]] = Field(
        None, description="Raw logits before activation"
    )

    # Confidence analysis
    confidence_metrics: Optional[ConfidenceMetrics] = Field(
        None, description="Advanced confidence measures"
    )
    uncertainty_score: Optional[float] = Field(
        None, description="Model uncertainty estimate"
    )

    # Statistical measures
    entropy: Optional[float] = Field(None, description="Prediction entropy")
    max_probability: Optional[float] = Field(
        None, description="Highest class probability"
    )
    probability_gap: Optional[float] = Field(
        None, description="Gap between top 2 predictions"
    )

    # Model interpretation
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )
    attention: Optional[AttentionData] = Field(
        None, description="Attention and visualization data"
    )

    # Artifact-specific analysis
    artifact_indicators: Optional[Dict[str, float]] = Field(
        None, description="Specific artifact indicators"
    )
    manipulation_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Manipulation detection confidence"
    )
    authenticity_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Detailed authenticity analysis"
    )

    # Processing metadata
    processing_time: Optional[float] = Field(
        None, description="Time to process this detection"
    )
    detection_model_version: Optional[str] = Field(
        None, description="Model version used"
    )
    detection_source: Optional[str] = Field(
        None, description="Which model/stage produced this detection"
    )


class ImageAnalysis(BaseModel):
    """Comprehensive image analysis metadata."""

    # Basic image properties
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    channels: int = Field(..., description="Number of channels")
    format: str = Field(..., description="Image format")
    file_size: Optional[int] = Field(None, description="File size in bytes")

    # Image quality metrics
    mean_brightness: Optional[float] = Field(None, description="Mean brightness value")
    contrast_ratio: Optional[float] = Field(None, description="Contrast ratio")
    color_distribution: Optional[Dict[str, float]] = Field(
        None, description="Color channel statistics"
    )
    histogram_stats: Optional[Dict[str, Any]] = Field(
        None, description="Histogram analysis"
    )

    # Technical metadata
    exif_data: Optional[Dict[str, Any]] = Field(
        None, description="EXIF metadata if available"
    )
    compression_ratio: Optional[float] = Field(
        None, description="Estimated compression ratio"
    )

    # Preprocessing applied
    preprocessing_steps: Optional[List[str]] = Field(
        None, description="Preprocessing steps applied"
    )
    resize_factor: Optional[float] = Field(None, description="Resize factor applied")


class ModelPerformanceMetrics(BaseModel):
    """Enhanced model performance metrics."""

    # Configure to avoid namespace conflicts
    model_config = {"protected_namespaces": ()}

    inference_time: float = Field(..., description="Total inference time in seconds")
    preprocessing_time: Optional[float] = Field(
        None, description="Image preprocessing time"
    )
    inference_forward_time: Optional[float] = Field(
        None, description="Model forward pass time"
    )
    postprocessing_time: Optional[float] = Field(
        None, description="Result postprocessing time"
    )

    # Memory usage
    peak_memory_usage: Optional[float] = Field(
        None, description="Peak memory usage in MB"
    )
    gpu_memory_usage: Optional[float] = Field(
        None, description="GPU memory usage in MB"
    )

    # Model statistics
    total_detections: Optional[int] = Field(
        None, description="Total number of detections"
    )
    filtered_detections: Optional[int] = Field(
        None, description="Detections after filtering"
    )
    nms_detections: Optional[int] = Field(None, description="Detections after NMS")

    # Confidence statistics
    mean_confidence: Optional[float] = Field(
        None, description="Mean detection confidence"
    )
    confidence_std: Optional[float] = Field(
        None, description="Confidence standard deviation"
    )
    high_confidence_count: Optional[int] = Field(
        None, description="Number of high-confidence detections"
    )

    # Processing details
    overall_confidence: Optional[float] = Field(
        None, description="Overall model confidence"
    )
    batch_size: Optional[int] = Field(None, description="Batch size used")
    device_used: Optional[str] = Field(None, description="Device used for inference")


class DetectionRequest(BaseModel):
    """Enhanced detection request parameters."""

    detection_type: DetectionType = Field(
        default=DetectionType.BOTH, description="Type of detection to perform"
    )
    confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence threshold for detections"
    )
    max_detections: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of detections to return",
    )

    # Enhanced options
    include_masks: bool = Field(default=False, description="Include segmentation masks")
    include_keypoints: bool = Field(
        default=False, description="Include keypoint detection"
    )
    include_features: bool = Field(
        default=False, description="Include raw model features"
    )
    include_attention: bool = Field(default=False, description="Include attention maps")
    return_top_k: int = Field(
        default=5, ge=1, le=20, description="Number of top predictions to return"
    )

    # Analysis options
    analyze_image_quality: bool = Field(
        default=True, description="Perform image quality analysis"
    )
    extract_metadata: bool = Field(default=True, description="Extract image metadata")


class DetectionResponse(BaseModel):
    """Enhanced detection response with comprehensive information."""

    success: bool = Field(..., description="Whether the detection was successful")
    message: str = Field(..., description="Response message")

    # Enhanced image analysis
    image_analysis: ImageAnalysis = Field(
        ..., description="Comprehensive image analysis"
    )

    # Detection results
    local_detections: Optional[List[LocalDetection]] = Field(
        default=None, description="Local artifact detections with enhanced info"
    )
    global_detection: Optional[GlobalDetection] = Field(
        default=None, description="Global artifact detection with detailed analysis"
    )

    # Performance and system info
    performance_metrics: ModelPerformanceMetrics = Field(
        ..., description="Performance metrics"
    )
    detection_type: DetectionType = Field(
        ..., description="Type of detection performed"
    )

    # Summary statistics
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary statistics")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    system_info: Dict[str, Any] = Field(..., description="System information")


class ErrorResponse(BaseModel):
    """Error response."""

    error: bool = Field(default=True, description="Error flag")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        default=None, description="Detailed error information"
    )
    error_code: Optional[str] = Field(default=None, description="Error code")


class InfoResponse(BaseModel):
    """Service information response."""

    # Configure to avoid namespace conflicts
    model_config = {"protected_namespaces": ()}

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    system_info: Dict[str, Any] = Field(..., description="System information")
    supported_formats: List[str] = Field(..., description="Supported image formats")
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    detection_types: List[str] = Field(..., description="Available detection types")
    detection_model_info: Dict[str, Any] = Field(..., description="Model information")

    # Enhanced capabilities
    supported_features: List[str] = Field(
        ..., description="Supported analysis features"
    )
    performance_benchmarks: Optional[Dict[str, float]] = Field(
        None, description="Model performance benchmarks"
    )


class ModelStatus(BaseModel):
    """Detailed status of a single model."""
    is_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_path: Optional[str] = Field(None, description="Path to the model weights file")
    model_size_mb: Optional[float] = Field(None, description="Size of the model in megabytes")
    simplified_mode: bool = Field(False, description="Whether the model is running in a simplified (fallback) mode")


class GPUInfo(BaseModel):
    """Detailed GPU information."""
    device_name: str = Field(..., description="GPU device name")
    total_memory_mb: float = Field(..., description="Total GPU memory in MB")
    used_memory_mb: float = Field(..., description="Used GPU memory in MB")
    free_memory_mb: float = Field(..., description="Free GPU memory in MB")
    active_memory_mb: float = Field(..., description="Active memory in MB (PyTorch)")
    allocated_memory_mb: float = Field(..., description="Allocated memory in MB (PyTorch)")
    reserved_memory_mb: float = Field(..., description="Reserved memory in MB (PyTorch)")


class ModelStatusResponse(BaseModel):
    """Response model for detailed model and GPU status."""
    model_status: Dict[str, ModelStatus] = Field(..., description="Status of each loaded model")
    gpu_info: Optional[GPUInfo] = Field(None, description="Information about the GPU, if available")
