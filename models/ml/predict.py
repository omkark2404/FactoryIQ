import os
import joblib
import pandas as pd
from app.config import RISK_MODEL_DIR

class QualityRiskPredictor:
    """Inference engine for manufacturing quality risk using saved Sklearn Pipelines."""
    def __init__(self):
        self.clf_pipeline = None
        self.reg_pipeline = None
        
        clf_path = RISK_MODEL_DIR / "risk_pipeline.joblib"
        reg_path = RISK_MODEL_DIR / "defect_pipeline.joblib"
        
        if clf_path.exists() and reg_path.exists():
            self.clf_pipeline = joblib.load(clf_path)
            self.reg_pipeline = joblib.load(reg_path)
        else:
            print("[QualityRiskPredictor] Warning: Models not found. Returning mock data.")

    def predict_risk(self, telemetry_dict: dict) -> dict:
        """Predicts rejection risk level and defect rate using fitted Sklearn Pipelines."""
        batch_id = telemetry_dict.get("batch_id", "UNKNOWN_BATCH")
        machine_id = telemetry_dict.get("machine_id", "UNKNOWN_MACHINE")
        
        if self.clf_pipeline is None or self.reg_pipeline is None:
            # Fallback mock response
            return {
                "batch_id": batch_id,
                "machine_id": machine_id,
                "risk_level": "HIGH",
                "rejection_risk_pct": 78.0,
                "predicted_defect_rate_pct": 2.52,
                "temperature_c": float(telemetry_dict.get("temperature_c", 184.0)),
                "shift": str(telemetry_dict.get("shift", "B"))
            }

        temp = float(telemetry_dict.get('temperature_c', 184.0))
        press = float(telemetry_dict.get('pressure_psi', 72.0))
        speed = float(telemetry_dict.get('line_speed_mmin', 45.0))
        shift_str = str(telemetry_dict.get('shift', 'B')).upper()
        prev_defects = int(telemetry_dict.get('previous_defects', 4))

        shift_encoded = 1 if shift_str == 'B' else (0 if shift_str == 'A' else 2)

        # Create single row dataframe
        df = pd.DataFrame([{
            'temperature_c': temp,
            'pressure_psi': press,
            'line_speed_mmin': speed,
            'shift_encoded': shift_encoded,
            'previous_defects': prev_defects
        }])

        prob_high_risk = self.clf_pipeline.predict_proba(df)[0][1]
        defect_rate = self.reg_pipeline.predict(df)[0]
        
        is_high = prob_high_risk > 0.5
        status = "HIGH" if is_high else "LOW"

        return {
            "batch_id": batch_id,
            "machine_id": machine_id,
            "risk_level": status,
            "rejection_risk_pct": round(prob_high_risk * 100, 1),
            "predicted_defect_rate_pct": round(defect_rate, 2),
            "temperature_c": temp,
            "shift": shift_str
        }
