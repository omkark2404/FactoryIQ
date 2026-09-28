import os
import glob

replacements = {
    'src.api': 'app',
    'src.rag': 'rag',
    'src.ml': 'models.ml',
    'src.vision': 'models.vision',
    'src.ocr': 'models.ocr',
    '"data/synthetic/production_data.csv"': 'os.path.join(str(DATA_DIR), "synthetic", "production_data.csv")',
    '"models/quality_risk_model"': 'str(RISK_MODEL_DIR)',
    '"models/vision_model/anomaly_detector.pth"': 'os.path.join(str(VISION_MODEL_DIR), "anomaly_detector.pth")',
    '"vector_store"': 'str(VECTOR_STORE_DIR)',
    '"data/raw/documents"': 'os.path.join(str(DATA_DIR), "raw", "documents")'
}

for root, _, files in os.walk('.'):
    if 'venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for f in files:
        if f.endswith('.py') and f != 'config.py' and f != 'update_imports.py':
            filepath = os.path.join(root, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace(old, new)
                
            # Add config import if we used config variables
            if 'DATA_DIR' in new_content or 'RISK_MODEL_DIR' in new_content or 'VISION_MODEL_DIR' in new_content or 'VECTOR_STORE_DIR' in new_content:
                if 'from app.config import' not in new_content:
                    new_content = 'from app.config import DATA_DIR, RISK_MODEL_DIR, VISION_MODEL_DIR, VECTOR_STORE_DIR\n' + new_content

            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Updated imports in {filepath}")
