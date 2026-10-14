import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Initialize FastAPI Test Client
client = TestClient(app)

def test_read_root():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "Supply Chain MLOps Inference Engine is active."}

def test_predict_sales_valid_payload():
    """Test the /predict endpoint with a valid payload."""
    payload = {
        "store_nbr": 1,
        "family": "GROCERY",
        "onpromotion": 1,
        "dcoilwtico": 45.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assert response structure and logical constraints
    assert data["store_nbr"] == 1
    assert data["family"] == "GROCERY"
    assert "predicted_sales" in data
    assert data["predicted_sales"] >= 0.0  # Crucial: Sales cannot be negative (RMSLE protection)
    assert data["model_used"] == "LightGBM_Ensemble_v1"

def test_predict_sales_invalid_payload():
    """Test the /predict endpoint with invalid data types to verify Pydantic validation."""
    payload = {
        "store_nbr": "invalid_store_string", # Should be an integer
        "family": "GROCERY",
        "onpromotion": 0
    }
    response = client.post("/predict", json=payload)
    # FastAPI should reject invalid data types with a 422 Unprocessable Entity error
    assert response.status_code == 422