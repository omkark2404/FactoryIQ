import os
import pandas as pd
import numpy as np
import urllib.request
import tarfile
from app.config import DATA_DIR, setup_seeds

def download_mvtec():
    """Downloads a small public sample of MVTec AD for demonstration."""
    # For a real pipeline, we download from https://www.mvtec.com/company/research/datasets/mvtec-ad
    # Since it is 5GB and requires registration, we use a placeholder or synthetic generator.
    print("[Data] MVTec AD dataset download requested.")
    print("[Data] NOTE: MVTec AD is CC BY-NC-SA 4.0. Ensure compliance for commercial use.")
    mvtec_dir = DATA_DIR / "raw" / "mvtec"
    os.makedirs(mvtec_dir, exist_ok=True)
    
    # We rely on the existing dummy local structure for this demo to avoid a 5GB download
    print(f"[Data] Using local structure at {mvtec_dir}")

def generate_synthetic_telemetry():
    """Generates synthetic production data for the ML risk model."""
    setup_seeds()
    out_dir = DATA_DIR / "synthetic"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "production_data.csv"
    
    n = 1000
    df = pd.DataFrame({
        'temperature_c': np.random.normal(170, 10, n),
        'pressure_psi': np.random.normal(70, 5, n),
        'line_speed_mmin': np.random.normal(50, 5, n),
        'shift': np.random.choice(['A','B','C'], n),
        'previous_defects': np.random.poisson(2, n)
    })
    # Inject logic: High temp -> High Risk
    df['quality_risk_level'] = np.where(df['temperature_c'] > 180, 'HIGH', 'LOW')
    df['defect_rate_pct'] = np.where(df['temperature_c'] > 180, np.random.uniform(2, 5, n), np.random.uniform(0, 1, n))
    
    df.to_csv(out_file, index=False)
    print(f"[Data] Synthetic telemetry data generated at {out_file}")

if __name__ == "__main__":
    print("=== FactoryIQ Data Setup ===")
    download_mvtec()
    generate_synthetic_telemetry()
    print("=== Setup Complete ===")
