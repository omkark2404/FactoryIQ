from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global dictionary to store ML models
ml_models = {}

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
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
