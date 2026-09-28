import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "bottle" in data["categories_supported"]

def test_api_vision_prediction():
    payload = {"category": "bottle", "force_defect": True}
    response = client.post("/api/predict/vision", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "ANOMALY"

def test_api_risk_prediction():
    payload = {
        "batch_id": "RB-2041",
        "machine_id": "Press-04",
        "temperature_c": 184.0,
        "shift": "B"
    }
    response = client.post("/api/predict/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "HIGH"

def test_api_batch_analysis():
    response = client.get("/api/analyze/batch/RB-2041?category=bottle")
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "RB-2041"
    assert data["risk_prediction"]["risk_level"] == "HIGH"
    assert len(data["rag_synthesis"]["sources"]) > 0
