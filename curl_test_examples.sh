#!/bin/bash
# HADM API Test Commands using curl
# Make sure your HADM server is running on localhost:8080

API_BASE="http://localhost:8080/api/v1"
IMAGE_FILE="test_cat.jpg"

echo "🚀 HADM API Test Commands"
echo "=========================="

# Download test image if it doesn't exist
if [ ! -f "$IMAGE_FILE" ]; then
    echo "📥 Downloading test cat image..."
    wget -O "$IMAGE_FILE" "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1024px-Cat_November_2010-1a.jpg"
    echo "✅ Test image downloaded: $IMAGE_FILE"
else
    echo "✅ Using existing test image: $IMAGE_FILE"
fi

echo ""
echo "🔍 1. Health Check:"
echo "curl -X GET \"$API_BASE/health\""
curl -X GET "$API_BASE/health" | jq '.' 2>/dev/null || curl -X GET "$API_BASE/health"

echo ""
echo ""
echo "📋 2. Service Info:"
echo "curl -X GET \"$API_BASE/info\""
curl -X GET "$API_BASE/info" | jq '.' 2>/dev/null || curl -X GET "$API_BASE/info"

echo ""
echo ""
echo "🔬 3. Local Detection (bounding boxes):"
echo "curl -X POST \"$API_BASE/detect/local\" -F \"image=@$IMAGE_FILE\" -F \"confidence_threshold=0.3\""
curl -X POST "$API_BASE/detect/local" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.' 2>/dev/null || \
curl -X POST "$API_BASE/detect/local" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50"

echo ""
echo ""
echo "🌍 4. Global Detection (whole image classification):"
echo "curl -X POST \"$API_BASE/detect/global\" -F \"image=@$IMAGE_FILE\" -F \"confidence_threshold=0.3\""
curl -X POST "$API_BASE/detect/global" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3" | jq '.' 2>/dev/null || \
curl -X POST "$API_BASE/detect/global" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3"

echo ""
echo ""
echo "🔄 5. Both Detection Types:"
echo "curl -X POST \"$API_BASE/detect/both\" -F \"image=@$IMAGE_FILE\" -F \"confidence_threshold=0.3\""
curl -X POST "$API_BASE/detect/both" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.' 2>/dev/null || \
curl -X POST "$API_BASE/detect/both" \
     -F "image=@$IMAGE_FILE" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50"

echo ""
echo ""
echo "🎯 6. Unified Detection Endpoint:"
echo "curl -X POST \"$API_BASE/detect\" -F \"image=@$IMAGE_FILE\" -F \"detection_type=both\" -F \"confidence_threshold=0.3\""
curl -X POST "$API_BASE/detect" \
     -F "image=@$IMAGE_FILE" \
     -F "detection_type=both" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50" | jq '.' 2>/dev/null || \
curl -X POST "$API_BASE/detect" \
     -F "image=@$IMAGE_FILE" \
     -F "detection_type=both" \
     -F "confidence_threshold=0.3" \
     -F "max_detections=50"

echo ""
echo ""
echo "🧹 Cleaning up test image..."
rm -f "$IMAGE_FILE"
echo "✅ Done!"

echo ""
echo "💡 Tips:"
echo "   - Install 'jq' for prettier JSON output: sudo apt install jq"
echo "   - Adjust confidence_threshold (0.0-1.0) to filter results"
echo "   - Use max_detections to limit number of bounding boxes"
echo "   - Check server logs if you encounter errors" 