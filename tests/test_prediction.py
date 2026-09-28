import pytest
from models.vision.predict import VisionPredictor
from models.ml.predict import QualityRiskPredictor

def test_vision_predictor():
    predictor = VisionPredictor()
    for cat in ["bottle", "screw", "metal_nut", "tile"]:
        res = predictor.predict_synthetic(category=cat, force_defect=True)
        assert res["category"] == cat
        assert res["result"] == "ANOMALY"
        assert res["anomaly_score"] >= 0.65

def test_ml_risk_predictor():
    predictor = QualityRiskPredictor()
    high_risk_input = {"batch_id": "RB-2041", "machine_id": "Press-04", "temperature_c": 184.5, "shift": "B"}
    res = predictor.predict_risk(high_risk_input)
    assert res["risk_level"] == "HIGH"
    assert res["rejection_risk_pct"] > 50.0

    low_risk_input = {"batch_id": "RB-2043", "machine_id": "Press-03", "temperature_c": 169.8, "shift": "C"}
    res_low = predictor.predict_risk(low_risk_input)
    assert res_low["risk_level"] == "LOW"
