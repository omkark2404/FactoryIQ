import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MLPreprocessor:
    """Preprocesses manufacturing telemetry data for quality risk training and inference."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.shift_encoder = LabelEncoder()
        self.feature_columns = [
            'temperature_c', 'pressure_psi', 'line_speed_mmin', 
            'shift_encoded', 'previous_defects'
        ]

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits scaler and encodes features from production DataFrame."""
        df_copy = df.copy()
        df_copy['shift_encoded'] = self.shift_encoder.fit_transform(df_copy['shift'].astype(str))
        
        X = df_copy[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)

        # Target 1: Risk Level (HIGH = 1, LOW = 0)
        y_risk = (df_copy['quality_risk_level'] == 'HIGH').astype(int).values
        
        # Target 2: Defect Rate %
        y_defect_rate = df_copy['defect_rate_pct'].values

        return X_scaled, y_risk, y_defect_rate

    def transform_single(self, sample_dict: dict) -> np.ndarray:
        """Transforms a single input sample dictionary into scaled feature vector."""
        temp = float(sample_dict.get('temperature_c', 184.0))
        press = float(sample_dict.get('pressure_psi', 72.0))
        speed = float(sample_dict.get('line_speed_mmin', 45.0))
        shift_str = str(sample_dict.get('shift', 'B')).upper()
        prev_defects = int(sample_dict.get('previous_defects', 4))

        shift_encoded = 1 if shift_str == 'B' else (0 if shift_str == 'A' else 2)

        raw_vec = np.array([[temp, press, speed, shift_encoded, prev_defects]])
        if hasattr(self.scaler, 'mean_'):
            return self.scaler.transform(raw_vec)
        return raw_vec
