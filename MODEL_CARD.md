# Model Card: FactoryIQ Vision & Telemetry Models

## 1. Vision Anomaly Detector
* **Architecture:** Convolutional Autoencoder (PyTorch)
* **Intended Use:** Detect surface defects (cracks, scratches) on industrial components.
* **Training Data:** MVTec AD Dataset (CC BY-NC-SA 4.0).
* **Metrics:** AUROC 0.94, F1 0.89 (Local Simulation).
* **Limitations:** Performance drops significantly on textures/categories not present in MVTec (e.g., highly reflective metals).

## 2. Telemetry Risk Predictor
* **Architecture:** Scikit-Learn RandomForestClassifier & RandomForestRegressor
* **Intended Use:** Predict batch rejection risk based on machine telemetry (temp, pressure, speed).
* **Training Data:** Fully synthetic `production_data.csv` generated via `scripts/download_data.py`.
* **Metrics:** Precision 1.0, Recall 1.0, F1 1.0 (on synthetic ruleset).
* **Limitations:** The model learns a synthetic deterministic rule (Temp > 180 = High Risk). It must be retrained on real factory sensor data before production use.
