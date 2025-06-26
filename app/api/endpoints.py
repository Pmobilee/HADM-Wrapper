"""
HADM Server API Endpoints
"""
import time
import logging
import platform
import psutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.hadm_models import model_manager
from app.models.schemas import (
    DetectionResponse,
    DetectionType,
    HealthResponse,
    InfoResponse,
    ErrorResponse
)
from app.utils.image_utils import load_image_from_upload, preprocess_for_hadm, postprocess_detections

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Get system information
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else 0
        }
        
        # Get model status
        models_loaded = model_manager.get_model_status()
        
        # Determine overall status
        status = "healthy" if any(models_loaded.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            models_loaded=models_loaded,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/info", response_model=InfoResponse)
async def get_info():
    """Get service information."""
    try:
        model_info = {
            "local_model": {
                "type": "HADM-L",
                "classes": 6,
                "description": "Local human artifact detection with bounding boxes"
            },
            "global_model": {
                "type": "HADM-G", 
                "classes": 12,
                "description": "Global human artifact detection for whole image"
            },
            "base_model": "EVA-02-L",
            "framework": "Detectron2"
        }
        
        return InfoResponse(
            name="HADM Server",
            version="1.0.0",
            description="FastAPI server for Human Artifact Detection in Machine-generated images",
            supported_formats=settings.supported_formats,
            max_file_size=settings.max_file_size,
            detection_types=["local", "global", "both"],
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Info request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service info")


@router.post("/detect/local", response_model=DetectionResponse)
async def detect_local_artifacts(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    max_detections: Optional[int] = Form(None, ge=1, le=1000)
):
    """
    Detect local human artifacts in the uploaded image.
    
    Returns bounding boxes for detected artifacts.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing local detection request for {image.filename}")
        
        # Load and validate image
        image_array, original_size = await load_image_from_upload(image)
        
        # Preprocess image
        processed_image, metadata = preprocess_for_hadm(image_array)
        
        # Run local detection
        local_detections = await model_manager.predict_local(processed_image)
        
        # Post-process detections to original coordinates
        local_detections = postprocess_detections(local_detections, metadata)
        
        # Apply custom thresholds if provided
        if confidence_threshold is not None:
            local_detections = [d for d in local_detections if d.confidence >= confidence_threshold]
        
        if max_detections is not None:
            local_detections = local_detections[:max_detections]
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            message=f"Local detection completed. Found {len(local_detections)} artifacts.",
            image_width=original_size[0],
            image_height=original_size[1],
            processed_width=settings.image_size,
            processed_height=settings.image_size,
            local_detections=local_detections,
            global_detection=None,
            processing_time=processing_time,
            model_version=model_manager.local_model.model_version,
            detection_type=DetectionType.LOCAL
        )
        
    except Exception as e:
        logger.error(f"Local detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Local detection failed: {str(e)}"
        )


@router.post("/detect/global", response_model=DetectionResponse)
async def detect_global_artifacts(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0)
):
    """
    Detect global human artifacts in the uploaded image.
    
    Returns classification result for the whole image.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing global detection request for {image.filename}")
        
        # Load and validate image
        image_array, original_size = await load_image_from_upload(image)
        
        # Preprocess image
        processed_image, metadata = preprocess_for_hadm(image_array)
        
        # Run global detection
        global_detection = await model_manager.predict_global(processed_image)
        
        # Apply custom threshold if provided
        if confidence_threshold is not None and global_detection:
            if global_detection.confidence < confidence_threshold:
                global_detection = None
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            message="Global detection completed.",
            image_width=original_size[0],
            image_height=original_size[1],
            processed_width=settings.image_size,
            processed_height=settings.image_size,
            local_detections=None,
            global_detection=global_detection,
            processing_time=processing_time,
            model_version=model_manager.global_model.model_version,
            detection_type=DetectionType.GLOBAL
        )
        
    except Exception as e:
        logger.error(f"Global detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Global detection failed: {str(e)}"
        )


@router.post("/detect/both", response_model=DetectionResponse)
async def detect_both_artifacts(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    max_detections: Optional[int] = Form(None, ge=1, le=1000)
):
    """
    Detect both local and global human artifacts in the uploaded image.
    
    Returns both bounding boxes and global classification.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing combined detection request for {image.filename}")
        
        # Load and validate image
        image_array, original_size = await load_image_from_upload(image)
        
        # Preprocess image
        processed_image, metadata = preprocess_for_hadm(image_array)
        
        # Run both detections
        local_detections, global_detection = await model_manager.predict_both(processed_image)
        
        # Post-process local detections
        local_detections = postprocess_detections(local_detections, metadata)
        
        # Apply custom thresholds if provided
        if confidence_threshold is not None:
            local_detections = [d for d in local_detections if d.confidence >= confidence_threshold]
            if global_detection and global_detection.confidence < confidence_threshold:
                global_detection = None
        
        if max_detections is not None:
            local_detections = local_detections[:max_detections]
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            message=f"Combined detection completed. Found {len(local_detections)} local artifacts.",
            image_width=original_size[0],
            image_height=original_size[1],
            processed_width=settings.image_size,
            processed_height=settings.image_size,
            local_detections=local_detections,
            global_detection=global_detection,
            processing_time=processing_time,
            model_version=f"L:{model_manager.local_model.model_version},G:{model_manager.global_model.model_version}",
            detection_type=DetectionType.BOTH
        )
        
    except Exception as e:
        logger.error(f"Combined detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Combined detection failed: {str(e)}"
        )


@router.post("/detect", response_model=DetectionResponse)
async def detect_artifacts(
    image: UploadFile = File(..., description="Image file to analyze"),
    detection_type: DetectionType = Form(DetectionType.BOTH, description="Type of detection to perform"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    max_detections: Optional[int] = Form(None, ge=1, le=1000)
):
    """
    Universal detection endpoint that supports all detection types.
    """
    if detection_type == DetectionType.LOCAL:
        return await detect_local_artifacts(image, confidence_threshold, max_detections)
    elif detection_type == DetectionType.GLOBAL:
        return await detect_global_artifacts(image, confidence_threshold)
    else:  # DetectionType.BOTH
        return await detect_both_artifacts(image, confidence_threshold, max_detections)


# Error handlers (these will be added to the main app in main.py)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            detail=str(exc),
            error_code="INTERNAL_ERROR"
        ).dict()
    ) 