import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")

# Feature/ML paths
VISION_MODEL_DIR = MODELS_DIR / "vision_model"
RISK_MODEL_DIR = MODELS_DIR / "quality_risk_model"

# API & LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 4))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

def setup_seeds(seed: int = RANDOM_SEED):
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
