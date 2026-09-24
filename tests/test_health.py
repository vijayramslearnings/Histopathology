import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health returns HTTP 200 and standard health payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "histopathology-ai-backend"
    assert data["version"] == "1.0.0"
    assert "disclaimer" in data
    assert "Not a clinical diagnostic system" in data["disclaimer"]
    assert "X-Research-Disclaimer" in response.headers


def test_model_health_endpoint():
    """Test GET /health/model returns model diagnostics."""
    response = client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert "device" in data
    assert data["checkpoint_status"] == "not_loaded"
    assert "disclaimer" in data


def test_api_v1_health_routes():
    """Test health endpoints mounted under /api/v1 prefix."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    response_model = client.get("/api/v1/health/model")
    assert response_model.status_code == 200
    assert response_model.json()["status"] == "ok"
