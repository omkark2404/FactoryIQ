from app.config import DATA_DIR, RISK_MODEL_DIR, VISION_MODEL_DIR, VECTOR_STORE_DIR
import os
import pickle
from models.ml.preprocess import MLPreprocessor

class QualityRiskPredictor:
    """Predictor engine for classical ML Quality Risk estimation."""
    def __init__(self, model_dir: str = str(RISK_MODEL_DIR)):
        self.model_dir = model_dir
        self.classifier = None
        self.regressor = None
        self.preprocessor = MLPreprocessor()
        
        clf_path = os.path.join(model_dir, "risk_classifier.pkl")
        reg_path = os.path.join(model_dir, "defect_regressor.pkl")
        prep_path = os.path.join(model_dir, "preprocessor.pkl")

        if os.path.exists(clf_path) and os.path.exists(reg_path) and os.path.exists(prep_path):
            try:
                with open(clf_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(reg_path, 'rb') as f:
                    self.regressor = pickle.load(f)
                with open(prep_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                print(f"[QualityRiskPredictor] Loaded ML models from {model_dir}")
            except Exception as e:
                print(f"[QualityRiskPredictor] Could not load ML artifacts ({e}). Using rule-based estimator.")

    def predict_risk(self, batch_data: dict) -> dict:
        """
        Predicts quality risk level (HIGH/LOW) and rejection probability %.
        Inputs: batch_data dict containing temperature_c, pressure_psi, shift, etc.
        """
        temp = float(batch_data.get("temperature_c", 184.0))
        batch_id = batch_data.get("batch_id", "RB-2041")
        machine_id = batch_data.get("machine_id", "Press-04")
        shift = batch_data.get("shift", "B")

        if self.classifier is not None and self.regressor is not None:
            X_scaled = self.preprocessor.transform_single(batch_data)
            risk_prob = float(self.classifier.predict_proba(X_scaled)[0][1])
            defect_rate = float(self.regressor.predict(X_scaled)[0])
        else:
            # Deterministic fallback estimator based on temperature thresholds
            if temp >= 180.0:
                risk_prob = min(0.95, 0.50 + (temp - 180.0) * 0.07)
                defect_rate = 1.5 + (temp - 180.0) * 0.25
            else:
                risk_prob = max(0.05, 0.20 - (180.0 - temp) * 0.01)
                defect_rate = max(0.2, 0.8 - (180.0 - temp) * 0.04)

        risk_level = "HIGH" if risk_prob >= 0.60 else "LOW"
        risk_pct = round(risk_prob * 100.0, 1)
        defect_rate_pct = round(defect_rate, 2)

        return {
            "batch_id": batch_id,
            "machine_id": machine_id,
            "shift": shift,
            "temperature_c": temp,
            "risk_level": risk_level,
            "rejection_risk_pct": risk_pct,
            "predicted_defect_rate_pct": defect_rate_pct,
            "recommendation": "⚠️ MANDATORY SAMPLING INSPECTION REQUIRED" if risk_level == "HIGH" else "✓ Normal production parameters"
        }
