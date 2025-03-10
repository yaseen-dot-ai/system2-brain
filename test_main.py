import pytest
from fastapi.testclient import TestClient
import base64
from PIL import Image
import io
from main import app, System2Request, System2Response

client = TestClient(app)

def create_test_image():
    """Create a simple test image and return its base64 encoding"""
    # Create a small blank image
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

def test_system2_endpoint_basic():
    """Test the /s2 endpoint with basic valid data"""
    test_data = {
        "task": "Click the submit button",
        "screenshot": create_test_image(),
        "context": {
            "current_url": "https://example.com",
            "page_title": "Test Page"
        }
    }
    
    response = client.post("/s2", json=test_data)
    assert response.status_code == 200
    
    response_data = response.json()
    assert "selected_element" in response_data
    assert "reasoning" in response_data
    assert "confidence" in response_data
    
    assert isinstance(response_data["selected_element"], dict)
    assert isinstance(response_data["reasoning"], str)
    assert isinstance(response_data["confidence"], float)

def test_system2_with_scratchpad():
    """Test the /s2 endpoint with scratchpad data"""
    test_data = {
        "task": "Click the submit button",
        "screenshot": create_test_image(),
        "scratchpad": "Previous thoughts: Looking for a blue button...",
        "context": {
            "current_url": "https://example.com",
            "page_title": "Test Page"
        }
    }
    
    response = client.post("/s2", json=test_data)
    assert response.status_code == 200
    
    response_data = response.json()
    assert "selected_element" in response_data
    assert "reasoning" in response_data
    assert "confidence" in response_data

def test_system2_invalid_base64():
    """Test the endpoint with invalid base64 data"""
    test_data = {
        "task": "Click the submit button",
        "screenshot": "invalid_base64_data",
        "context": {
            "current_url": "https://example.com",
            "page_title": "Test Page"
        }
    }
    
    response = client.post("/s2", json=test_data)
    assert response.status_code == 400
    
def test_system2_missing_fields():
    """Test the endpoint with missing required fields"""
    test_data = {
        # Missing required fields
        "context": {}
    }
    
    response = client.post("/s2", json=test_data)
    assert response.status_code == 422  # FastAPI validation error 