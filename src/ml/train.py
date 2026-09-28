import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.ml.preprocess import MLPreprocessor

def train_quality_risk_model(
    data_path: str = "data/synthetic/production_data.csv",
    save_dir: str = "models/quality_risk_model"
):
    """Trains classical ML model to predict quality rejection risk and defect rate."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"[ML Train] Loading production dataset from '{data_path}'...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Production data file not found at {data_path}")

    df = pd.read_csv(data_path)
    preprocessor = MLPreprocessor()
    X_scaled, y_risk, y_defect_rate = preprocessor.fit_transform(df)

    # Train Classifier for Quality Risk (HIGH / LOW)
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(X_scaled, y_risk)

    # Train Regressor for Defect Rate %
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    regressor.fit(X_scaled, y_defect_rate)

    # Save artifacts
    clf_path = os.path.join(save_dir, "risk_classifier.pkl")
    reg_path = os.path.join(save_dir, "defect_regressor.pkl")
    prep_path = os.path.join(save_dir, "preprocessor.pkl")

    with open(clf_path, 'wb') as f:
        pickle.dump(classifier, f)
    with open(reg_path, 'wb') as f:
        pickle.dump(regressor, f)
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"[ML Train] ML Risk Models trained successfully and saved to '{save_dir}'")
    return save_dir

if __name__ == "__main__":
    train_quality_risk_model()
