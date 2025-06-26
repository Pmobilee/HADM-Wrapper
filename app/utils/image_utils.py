"""
Image processing utilities for HADM Server
"""

import io
import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image, ImageOps
import cv2
from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import ImageAnalysis

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
    if hasattr(file, "size") and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size} bytes",
        )

    # Check file extension
    if file.filename:
        ext = file.filename.lower().split(".")[-1]
        if ext not in settings.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.supported_formats)}",
            )

    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")


async def load_image_from_upload(
    file: UploadFile,
) -> Tuple[np.ndarray, Tuple[int, int], str]:
    """
    Load image from uploaded file.

    Args:
        file: Uploaded file object

    Returns:
        Tuple of (image_array, original_dimensions, format)

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

        # Detect format from filename or content
        detected_format = "JPEG"  # Default
        if file.filename:
            ext = file.filename.lower().split(".")[-1]
            if ext in ["png"]:
                detected_format = "PNG"
            elif ext in ["jpg", "jpeg"]:
                detected_format = "JPEG"
            elif ext in ["gif"]:
                detected_format = "GIF"
            elif ext in ["bmp"]:
                detected_format = "BMP"
            elif ext in ["webp"]:
                detected_format = "WEBP"

        # Load image with PIL - enhanced approach
        try:
            # First, try direct loading from BytesIO
            pil_image = Image.open(image_bytes)
            pil_image.load()  # Force loading of image data
            
            # Get actual format from PIL if available
            if pil_image.format:
                detected_format = pil_image.format
                
            logger.info(
                f"Successfully loaded image directly from BytesIO: {pil_image.size}, mode: {pil_image.mode}, format: {detected_format}"
            )

        except Exception as img_error:
            logger.error(f"PIL failed to open image from BytesIO: {img_error}")
            logger.info("Trying alternative approach with temporary file...")

            # Try with different approach - save to temp file
            import tempfile

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    temp_path = temp_file.name

                pil_image = Image.open(temp_path)
                pil_image.load()  # Force loading
                
                # Get actual format from PIL if available
                if pil_image.format:
                    detected_format = pil_image.format
                    
                logger.info(
                    f"Successfully loaded image using temporary file: {pil_image.size}, mode: {pil_image.mode}, format: {detected_format}"
                )

                # Clean up temp file
                os.unlink(temp_path)

            except Exception as temp_error:
                logger.error(
                    f"Failed to load image even with temporary file: {temp_error}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot process image file: {str(temp_error)}",
                )

        # Get original dimensions
        original_size = pil_image.size  # (width, height)

        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array
        image_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV compatibility
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        logger.info(
            f"Loaded image: {original_size[0]}x{original_size[1]} pixels, mode: {pil_image.mode}, format: {detected_format}"
        )

        return image_array, original_size, detected_format

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        logger.error(
            f"File info - name: {file.filename}, content_type: {file.content_type}"
        )
        raise HTTPException(
            status_code=400, detail=f"Failed to process image: {str(e)}"
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
        pil_image.save(buffer, format="JPEG", quality=quality)

        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to convert image to JPEG: {e}")
        raise


def resize_image(
    image: np.ndarray, target_size: int = 1024, maintain_aspect: bool = True
) -> np.ndarray:
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


def pad_image_square(
    image: np.ndarray,
    target_size: int = 1024,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
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
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
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
        "original_size": (original_w, original_h),
        "processed_size": (settings.image_size, settings.image_size),
        "scale_factor": settings.image_size / max(original_h, original_w),
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
    if not detections or "scale_factor" not in metadata:
        return detections

    scale_factor = metadata["scale_factor"]
    original_w, original_h = metadata["original_size"]

    # Calculate padding offsets
    processed_h = int(original_h * scale_factor)
    processed_w = int(original_w * scale_factor)
    pad_h = (settings.image_size - processed_h) // 2
    pad_w = (settings.image_size - processed_w) // 2

    processed_detections = []
    for detection in detections:
        if hasattr(detection, "bbox"):
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


def create_visualization(
    image: np.ndarray, local_detections: list = None, global_detection=None
) -> np.ndarray:
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
                2,
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
                1,
            )

    # Draw global detection info
    if global_detection:
        label = (
            f"Global: {global_detection.class_name} ({global_detection.confidence:.2f})"
        )
        cv2.putText(
            vis_image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red color
            2,
        )

    return vis_image


def save_image(image: np.ndarray, filepath: str, format: str = "JPEG") -> bool:
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


def analyze_image(image: np.ndarray, file_format: str = "JPEG") -> ImageAnalysis:
    """
    Analyze image properties and metadata.

    Args:
        image: Input image as numpy array
        file_format: Image format (default: JPEG)

    Returns:
        ImageAnalysis object containing image analysis data
    """
    try:
        # Basic image properties
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        # Calculate file size estimation (in memory)
        file_size = image.nbytes

        # Determine color space
        color_space = (
            "RGB"
            if channels == 3
            else "Grayscale" if channels == 1 else f"{channels}-channel"
        )

        # Calculate basic statistics
        mean_brightness = float(np.mean(image))
        std_brightness = float(np.std(image))

        # Calculate image quality metrics
        # Sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if channels == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        # Contrast using standard deviation
        contrast = float(np.std(gray))

        # Noise estimation (simplified)
        noise_estimate = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))

        # Color distribution (for RGB images)
        color_distribution = {}
        if channels == 3:
            color_distribution = {
                "blue_mean": float(np.mean(image[:, :, 0])),
                "green_mean": float(np.mean(image[:, :, 1])),
                "red_mean": float(np.mean(image[:, :, 2])),
                "blue_std": float(np.std(image[:, :, 0])),
                "green_std": float(np.std(image[:, :, 1])),
                "red_std": float(np.std(image[:, :, 2])),
            }

        # Aspect ratio
        aspect_ratio = float(width / height)

        # Determine image category by size
        total_pixels = width * height
        if total_pixels < 500 * 500:
            size_category = "small"
        elif total_pixels < 1920 * 1080:
            size_category = "medium"
        else:
            size_category = "large"

        # Create and return ImageAnalysis object
        return ImageAnalysis(
            width=width,
            height=height,
            channels=channels,
            format=file_format,
            file_size=file_size,
            mean_brightness=mean_brightness,
            contrast_ratio=contrast,
            color_distribution=color_distribution,
            histogram_stats={
                "mean_brightness": mean_brightness,
                "std_brightness": std_brightness,
                "sharpness": sharpness,
                "noise_estimate": noise_estimate,
                "aspect_ratio": aspect_ratio,
                "size_category": size_category,
                "total_pixels": total_pixels,
                "color_space": color_space,
            }
        )

    except Exception as e:
        logger.warning(f"Error analyzing image: {e}")
        # Return minimal analysis on error
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return ImageAnalysis(
            width=width,
            height=height,
            channels=channels,
            format=file_format,
            file_size=image.nbytes,
            mean_brightness=float(np.mean(image)) if image.size > 0 else 0.0,
            contrast_ratio=float(np.std(image)) if image.size > 0 else 0.0,
        )


def visualize_detections(
    image: np.ndarray,
    local_detections: Optional[List] = None,
    global_detection: Optional[Any] = None,
    confidence_threshold: float = 0.5,
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Create visualization of detection results on the image.

    Args:
        image: Input image as numpy array
        local_detections: List of local detection objects
        global_detection: Global detection object
        confidence_threshold: Minimum confidence to visualize
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        line_thickness: Thickness of bounding box lines

    Returns:
        Image with visualizations drawn
    """
    try:
        # Create a copy to avoid modifying original
        vis_image = image.copy()

        # Define colors for different classes (BGR format for OpenCV)
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]

        # Draw local detections (bounding boxes)
        if local_detections:
            for i, detection in enumerate(local_detections):
                confidence = getattr(detection, "confidence", 0.0)

                # Skip low confidence detections
                if confidence < confidence_threshold:
                    continue

                # Get bounding box
                bbox = getattr(detection, "bbox", None)
                if bbox:
                    x1 = int(getattr(bbox, "x1", 0))
                    y1 = int(getattr(bbox, "y1", 0))
                    x2 = int(getattr(bbox, "x2", 0))
                    y2 = int(getattr(bbox, "y2", 0))

                    # Choose color based on class or index
                    color = colors[i % len(colors)]

                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)

                    # Prepare label text
                    label_parts = []
                    if show_labels:
                        class_name = getattr(detection, "class_name", f"class_{i}")
                        label_parts.append(class_name)

                    if show_confidence:
                        label_parts.append(f"{confidence:.2f}")

                    if label_parts:
                        label = " ".join(label_parts)

                        # Calculate text size and position
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 1

                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, text_thickness
                        )

                        # Draw background rectangle for text
                        cv2.rectangle(
                            vis_image,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            color,
                            -1,
                        )

                        # Draw text
                        cv2.putText(
                            vis_image,
                            label,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (255, 255, 255),  # White text
                            text_thickness,
                        )

        # Draw global detection info (as overlay text)
        if global_detection:
            confidence = getattr(global_detection, "confidence", 0.0)

            if confidence >= confidence_threshold:
                class_name = getattr(global_detection, "class_name", "unknown")

                # Create global detection text
                global_text = f"Global: {class_name}"
                if show_confidence:
                    global_text += f" ({confidence:.2f})"

                # Position at top-left corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_thickness = 2

                (text_width, text_height), baseline = cv2.getTextSize(
                    global_text, font, font_scale, text_thickness
                )

                # Draw background rectangle
                cv2.rectangle(
                    vis_image,
                    (10, 10),
                    (20 + text_width, 20 + text_height),
                    (0, 0, 0),  # Black background
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    global_text,
                    (15, 15 + text_height),
                    font,
                    font_scale,
                    (0, 255, 255),  # Yellow text
                    text_thickness,
                )

        # Add detection count info
        if local_detections:
            valid_detections = len(
                [
                    d
                    for d in local_detections
                    if getattr(d, "confidence", 0) >= confidence_threshold
                ]
            )
            count_text = f"Detections: {valid_detections}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                count_text, font, font_scale, text_thickness
            )

            # Position at bottom-right
            img_height, img_width = vis_image.shape[:2]
            x_pos = img_width - text_width - 15
            y_pos = img_height - 15

            # Draw background
            cv2.rectangle(
                vis_image,
                (x_pos - 5, y_pos - text_height - 5),
                (x_pos + text_width + 5, y_pos + 5),
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                vis_image,
                count_text,
                (x_pos, y_pos),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
            )

        return vis_image

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        # Return original image on error
        return image.copy()
