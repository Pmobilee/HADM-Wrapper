"""
HADM Server API Schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DetectionType(str, Enum):
    """Detection type enumeration."""
    LOCAL = "local"
    GLOBAL = "global"
    BOTH = "both"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height


class LocalDetection(BaseModel):
    """Local artifact detection result."""
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    class_id: int = Field(..., description="Detected class ID")
    class_name: str = Field(..., description="Detected class name")


class GlobalDetection(BaseModel):
    """Global artifact detection result."""
    class_id: int = Field(..., description="Predicted class ID")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    probabilities: Dict[str, float] = Field(..., description="All class probabilities")


class DetectionRequest(BaseModel):
    """Detection request parameters."""
    detection_type: DetectionType = Field(default=DetectionType.BOTH, description="Type of detection to perform")
    confidence_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence threshold for detections"
    )
    max_detections: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=1000, 
        description="Maximum number of detections to return"
    )


class DetectionResponse(BaseModel):
    """Detection response."""
    success: bool = Field(..., description="Whether the detection was successful")
    message: str = Field(..., description="Response message")
    
    # Image metadata
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    processed_width: int = Field(..., description="Processed image width")
    processed_height: int = Field(..., description="Processed image height")
    
    # Detection results
    local_detections: Optional[List[LocalDetection]] = Field(
        default=None, 
        description="Local artifact detections"
    )
    global_detection: Optional[GlobalDetection] = Field(
        default=None, 
        description="Global artifact detection"
    )
    
    # Processing metadata
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    detection_type: DetectionType = Field(..., description="Type of detection performed")


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
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    error_code: Optional[str] = Field(default=None, description="Error code")


class InfoResponse(BaseModel):
    """Service information response."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Service description")
    supported_formats: List[str] = Field(..., description="Supported image formats")
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    detection_types: List[str] = Field(..., description="Supported detection types")
    model_info: Dict[str, Any] = Field(..., description="Model information") 