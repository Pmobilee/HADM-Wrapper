"""
Image processing utilities for HADM Server
"""
import io
import os
import logging
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageOps
import cv2
from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)


def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file.
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size} bytes"
        )
    
    # Check file extension
    if file.filename:
        ext = file.filename.lower().split('.')[-1]
        if ext not in settings.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.supported_formats)}"
            )
    
    # Check content type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )


async def load_image_from_upload(file: UploadFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load image from uploaded file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (image_array, original_dimensions)
        
    Raises:
        HTTPException: If image loading fails
    """
    try:
        # Validate file
        validate_image_file(file)
        
        # Read file content
        content = await file.read()
        
        # Create BytesIO object and ensure it's at the beginning
        image_bytes = io.BytesIO(content)
        image_bytes.seek(0)  # Ensure we're at the beginning
        
        # Load image with PIL - enhanced approach
        try:
            # First, try direct loading from BytesIO
            pil_image = Image.open(image_bytes)
            pil_image.load()  # Force loading of image data
            logger.info(f"Successfully loaded image directly from BytesIO: {pil_image.size}, mode: {pil_image.mode}")
            
        except Exception as img_error:
            logger.error(f"PIL failed to open image from BytesIO: {img_error}")
            logger.info("Trying alternative approach with temporary file...")
            
            # Try with different approach - save to temp file
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    temp_path = temp_file.name
                
                pil_image = Image.open(temp_path)
                pil_image.load()  # Force loading
                logger.info(f"Successfully loaded image using temporary file: {pil_image.size}, mode: {pil_image.mode}")
                
                # Clean up temp file
                os.unlink(temp_path)
                
            except Exception as temp_error:
                logger.error(f"Failed to load image even with temporary file: {temp_error}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot process image file: {str(temp_error)}"
                )
        
        # Get original dimensions
        original_size = pil_image.size  # (width, height)
        
        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV compatibility
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Loaded image: {original_size[0]}x{original_size[1]} pixels, mode: {pil_image.mode}")
        
        return image_array, original_size
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        logger.error(f"File info - name: {file.filename}, content_type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


def convert_to_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    """
    Convert image to JPEG format (required by HADM models).
    
    Args:
        image: Image as numpy array (BGR format)
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG image as bytes
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Save as JPEG to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to convert image to JPEG: {e}")
        raise


def resize_image(image: np.ndarray, target_size: int = 1024, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for the longer dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scale to fit target size
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = target_size, target_size
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized


def pad_image_square(image: np.ndarray, target_size: int = 1024, pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Pad image to square shape.
    
    Args:
        image: Input image as numpy array
        target_size: Target square size
        pad_color: RGB color for padding
        
    Returns:
        Square padded image
    """
    h, w = image.shape[:2]
    
    # Calculate padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    # Pad image
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=pad_color
    )
    
    return padded


def preprocess_for_hadm(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Preprocess image for HADM models.
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    original_h, original_w = image.shape[:2]
    
    # Resize maintaining aspect ratio
    resized = resize_image(image, settings.image_size, maintain_aspect=True)
    
    # Pad to square
    processed = pad_image_square(resized, settings.image_size)
    
    # Create metadata
    metadata = {
        'original_size': (original_w, original_h),
        'processed_size': (settings.image_size, settings.image_size),
        'scale_factor': settings.image_size / max(original_h, original_w)
    }
    
    return processed, metadata


def postprocess_detections(detections: list, metadata: dict) -> list:
    """
    Post-process detection results to original image coordinates.
    
    Args:
        detections: List of detection results
        metadata: Processing metadata from preprocess_for_hadm
        
    Returns:
        Detections with coordinates scaled back to original image
    """
    if not detections or 'scale_factor' not in metadata:
        return detections
    
    scale_factor = metadata['scale_factor']
    original_w, original_h = metadata['original_size']
    
    # Calculate padding offsets
    processed_h = int(original_h * scale_factor)
    processed_w = int(original_w * scale_factor)
    pad_h = (settings.image_size - processed_h) // 2
    pad_w = (settings.image_size - processed_w) // 2
    
    processed_detections = []
    for detection in detections:
        if hasattr(detection, 'bbox'):
            # Adjust bounding box coordinates
            bbox = detection.bbox
            
            # Remove padding offset
            x1 = (bbox.x1 - pad_w) / scale_factor
            y1 = (bbox.y1 - pad_h) / scale_factor
            x2 = (bbox.x2 - pad_w) / scale_factor
            y2 = (bbox.y2 - pad_h) / scale_factor
            
            # Clamp to image bounds
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            # Update detection
            detection.bbox.x1 = x1
            detection.bbox.y1 = y1
            detection.bbox.x2 = x2
            detection.bbox.y2 = y2
        
        processed_detections.append(detection)
    
    return processed_detections


def create_visualization(image: np.ndarray, local_detections: list = None, global_detection = None) -> np.ndarray:
    """
    Create visualization of detection results.
    
    Args:
        image: Input image as numpy array
        local_detections: List of local detections
        global_detection: Global detection result
        
    Returns:
        Visualization image
    """
    vis_image = image.copy()
    
    # Draw local detections
    if local_detections:
        for detection in local_detections:
            bbox = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),  # Green color
                2
            )
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
    
    # Draw global detection info
    if global_detection:
        label = f"Global: {global_detection.class_name} ({global_detection.confidence:.2f})"
        cv2.putText(
            vis_image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red color
            2
        )
    
    return vis_image


def save_image(image: np.ndarray, filepath: str, format: str = 'JPEG') -> bool:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array (BGR format)
        filepath: Output file path
        format: Image format
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and save
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(filepath, format=format)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False 