# HADM API Testing Guide

This guide provides step-by-step instructions for testing the HADM Server API with a sample cat image.

## Prerequisites

- HADM Server running on `http://localhost:8080`
- `curl` command-line tool installed
- `wget` for downloading test images
- Optional: `jq` for pretty JSON formatting (`sudo apt install jq`)

## Quick Start

### Option 1: Automated Testing (Recommended)

Run the comprehensive Python test script:

```bash
python3 test_api.py
```

This script will:
- Download a test cat image automatically
- Test all API endpoints
- Show detailed results
- Clean up test files

### Option 2: Shell Script Testing

Make the curl test script executable and run it:

```bash
chmod +x curl_test_examples.sh
./curl_test_examples.sh
```

### Option 3: Manual Testing

Follow the manual steps below for individual endpoint testing.

## Manual Testing Steps

### Step 1: Download Test Image

```bash
wget -O test_cat.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1024px-Cat_November_2010-1a.jpg"
```

### Step 2: Test Health Endpoint

```bash
curl -X GET "http://localhost:8080/api/v1/health" | jq '.'
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "hadm_local": true,
    "hadm_global": true
  },
  "system_info": {...}
}
```

### Step 3: Test Service Info

```bash
curl -X GET "http://localhost:8080/api/v1/info" | jq '.'
```

Expected response:
```json
{
  "name": "HADM Server",
  "version": "1.0.0",
  "description": "FastAPI server for Human Artifact Detection in Machine-generated images",
  "supported_formats": ["jpg", "jpeg", "png", "webp"],
  "max_file_size": 10485760,
  "detection_types": ["local", "global", "both"],
  "model_info": {...}
}
```

### Step 4: Test Local Detection (Bounding Boxes)

```bash
curl -X POST "http://localhost:8080/api/v1/detect/local" \
     -F "image=@test_cat.jpg" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.'
```

Expected response:
```json
{
  "success": true,
  "message": "Local detection completed. Found X artifacts.",
  "image_width": 1024,
  "image_height": 768,
  "processed_width": 1024,
  "processed_height": 1024,
  "local_detections": [
    {
      "bbox": {
        "x1": 100.0,
        "y1": 150.0,
        "x2": 300.0,
        "y2": 400.0
      },
      "confidence": 0.85,
      "class_id": 1,
      "class_name": "artifact_class_name"
    }
  ],
  "global_detection": null,
  "processing_time": 0.234,
  "model_version": "HADM-L_0249999",
  "detection_type": "local"
}
```

### Step 5: Test Global Detection (Whole Image Classification)

```bash
curl -X POST "http://localhost:8080/api/v1/detect/global" \
     -F "image=@test_cat.jpg" \
     -F "confidence_threshold=0.3" | jq '.'
```

Expected response:
```json
{
  "success": true,
  "message": "Global detection completed.",
  "image_width": 1024,
  "image_height": 768,
  "processed_width": 1024,
  "processed_height": 1024,
  "local_detections": null,
  "global_detection": {
    "class_id": 2,
    "class_name": "global_artifact_class",
    "confidence": 0.92,
    "probabilities": {
      "class_1": 0.05,
      "class_2": 0.92,
      "class_3": 0.03
    }
  },
  "processing_time": 0.187,
  "model_version": "HADM-G_0249999",
  "detection_type": "global"
}
```

### Step 6: Test Both Detection Types

```bash
curl -X POST "http://localhost:8080/api/v1/detect/both" \
     -F "image=@test_cat.jpg" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.'
```

### Step 7: Test Unified Detection Endpoint

```bash
curl -X POST "http://localhost:8080/api/v1/detect" \
     -F "image=@test_cat.jpg" \
     -F "detection_type=both" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.'
```

### Step 8: Clean Up

```bash
rm test_cat.jpg
```

## API Parameters

### Common Parameters

- `image`: Image file to analyze (required)
- `confidence_threshold`: Minimum confidence score (0.0-1.0, optional)
- `max_detections`: Maximum number of detections (1-1000, optional, local only)

### Detection Types

- `local`: Returns bounding boxes for detected artifacts
- `global`: Returns whole-image classification
- `both`: Returns both local and global detection results

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure HADM server is running on localhost:8080
   - Check server logs for startup errors

2. **Model Loading Errors**
   - Verify model files exist in `pretrained_models/` directory
   - Check model file permissions
   - Ensure sufficient GPU/CPU memory

3. **Image Format Errors**
   - Supported formats: JPEG, PNG, WebP
   - Maximum file size: 10MB
   - Ensure image is not corrupted

4. **Low Detection Results**
   - Try lowering confidence_threshold (e.g., 0.1)
   - Check if image contains detectable artifacts
   - Verify model is loaded correctly

### Server Logs

Check server logs for detailed error information:

```bash
tail -f logs/hadm_server.log
```

### Health Check

Always start troubleshooting with a health check:

```bash
curl -X GET "http://localhost:8080/api/v1/health"
```

## Expected Behavior

- **Cat Image**: Since we're testing with a natural cat photo, the HADM models (designed for detecting human artifacts in machine-generated images) should return low confidence scores or no detections
- **Processing Time**: Typical processing times range from 0.1-2.0 seconds depending on hardware
- **Response Format**: All responses follow the standardized JSON schema

## Next Steps

Once basic testing is complete:

1. Test with machine-generated images for better detection results
2. Experiment with different confidence thresholds
3. Test batch processing capabilities
4. Implement error handling in your client applications
5. Set up monitoring and logging for production use 