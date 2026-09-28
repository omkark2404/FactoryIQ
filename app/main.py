import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global dictionary to store ML models
ml_models = {}

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models globally
    print("[FastAPI] Loading Machine Learning models into memory...")
    from models.vision.predict import VisionPredictor
    from models.ml.predict import QualityRiskPredictor
    from rag.pipeline import RAGPipeline
    
    ml_models["vision_predictor"] = VisionPredictor()
    ml_models["risk_predictor"] = QualityRiskPredictor()
    ml_models["rag_pipeline"] = RAGPipeline()
    print("[FastAPI] Models loaded successfully.")
    
    yield
    
    # Shutdown
    print("[FastAPI] Shutting down and cleaning up models...")
    ml_models.clear()

app = FastAPI(
    title="FactoryIQ API",
    description="Manufacturing Intelligence Backend",
    version="2.0.0",
    lifespan=lifespan,
    debug=os.getenv("DEBUG", "False").lower() == "true",
    dependencies=[Depends(get_api_key)] if API_KEY else []
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from app.routes import router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
