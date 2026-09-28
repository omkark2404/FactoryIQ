import os
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from app.config import DATA_DIR, RISK_MODEL_DIR

def train_quality_risk_model():
    """Trains classical ML model to predict quality rejection risk and defect rate."""
    os.makedirs(RISK_MODEL_DIR, exist_ok=True)
    data_path = DATA_DIR / "synthetic" / "production_data.csv"
    print(f"[ML Train] Loading production dataset from '{data_path}'...")

    if not data_path.exists():
        raise FileNotFoundError(f"Production data file not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Feature Engineering inside pandas before sklearn pipeline for simplicity
    df['shift_encoded'] = df['shift'].map({'A': 0, 'B': 1, 'C': 2}).fillna(0)
    
    # Sort by time to ensure time-based split is valid (assuming dataframe order is temporal)
    # If there is a timestamp column, we would sort by it. Here we assume index is time.
    
    feature_columns = ['temperature_c', 'pressure_psi', 'line_speed_mmin', 'shift_encoded', 'previous_defects']
    X = df[feature_columns]
    
    y_risk = (df['quality_risk_level'] == 'HIGH').astype(int)
    y_defect_rate = df['defect_rate_pct']

    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_risk_train, y_risk_test = y_risk.iloc[:split_idx], y_risk.iloc[split_idx:]
    y_reg_train, y_reg_test = y_defect_rate.iloc[:split_idx], y_defect_rate.iloc[split_idx:]

    # Handle Class Imbalance for Classifier
    import numpy as np
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_risk_train)
    class_weight = {0: weights[0], 1: weights[1]}

    # Create Sklearn Pipelines
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, class_weight=class_weight))
    ])

    reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    print("[ML Train] Performing TimeSeries Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(clf_pipeline, X_train, y_risk_train, cv=tscv, scoring='roc_auc')
    print(f"[ML Train] CV ROC-AUC Scores: {cv_scores}, Mean: {cv_scores.mean():.4f}")

    print("[ML Train] Fitting final pipelines...")
    clf_pipeline.fit(X_train, y_risk_train)
    reg_pipeline.fit(X_train, y_reg_train)
    
    # Evaluate on test set
    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n[ML Train] Test Set Classification Report:")
    print(classification_report(y_risk_test, y_pred))
    print(f"Test ROC-AUC: {roc_auc_score(y_risk_test, y_proba):.4f}")

    # Save artifacts with joblib
    clf_path = RISK_MODEL_DIR / "risk_pipeline.joblib"
    reg_path = RISK_MODEL_DIR / "defect_pipeline.joblib"

    joblib.dump(clf_pipeline, clf_path)
    joblib.dump(reg_pipeline, reg_path)

    print(f"[ML Train] ML Risk Models trained successfully and saved to '{RISK_MODEL_DIR}'")

if __name__ == "__main__":
    train_quality_risk_model()
