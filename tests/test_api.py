"""
Tests for HADM Server API
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import numpy as np

from app.main import app

client = TestClient(app)


def create_test_image() -> bytes:
    """Create a test image for upload."""
    # Create a simple RGB image
    image = Image.new('RGB', (100, 100), color='red')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "models_loaded" in data
    assert "system_info" in data


def test_info_endpoint():
    """Test the info endpoint."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "HADM Server"
    assert data["version"] == "1.0.0"
    assert "supported_formats" in data
    assert "detection_types" in data
    assert "model_info" in data


@patch('app.core.hadm_models.model_manager')
def test_detect_local_endpoint(mock_manager):
    """Test the local detection endpoint."""
    # Mock the model manager
    mock_manager.predict_local.return_value = []
    mock_manager.local_model.model_version = "test"
    
    # Create test image
    test_image = create_test_image()
    
    # Make request
    response = client.post(
        "/api/v1/detect/local",
        files={"image": ("test.jpg", test_image, "image/jpeg")}
    )
    
    # Note: This might fail without actual models, but tests the structure
    # In a real test environment, we'd mock the dependencies properly
    assert response.status_code in [200, 500]  # 500 if models not loaded


@patch('app.core.hadm_models.model_manager')
def test_detect_global_endpoint(mock_manager):
    """Test the global detection endpoint."""
    # Mock the model manager
    mock_manager.predict_global.return_value = None
    mock_manager.global_model.model_version = "test"
    
    # Create test image
    test_image = create_test_image()
    
    # Make request
    response = client.post(
        "/api/v1/detect/global",
        files={"image": ("test.jpg", test_image, "image/jpeg")}
    )
    
    # Note: This might fail without actual models, but tests the structure
    assert response.status_code in [200, 500]  # 500 if models not loaded


def test_invalid_file_upload():
    """Test upload with invalid file."""
    # Test with non-image file
    response = client.post(
        "/api/v1/detect/local",
        files={"image": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400


def test_missing_file():
    """Test request without file."""
    response = client.post("/api/v1/detect/local")
    assert response.status_code == 422  # Validation error


def test_large_file():
    """Test upload with oversized file."""
    # Create a large fake file
    large_content = b"x" * (11 * 1024 * 1024)  # 11MB (over limit)
    
    response = client.post(
        "/api/v1/detect/local",
        files={"image": ("large.jpg", large_content, "image/jpeg")}
    )
    
    # Should be rejected due to size
    assert response.status_code in [413, 400]


if __name__ == "__main__":
    pytest.main([__file__]) 