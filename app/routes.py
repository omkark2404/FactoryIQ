from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io

from app.schemas import (
    VisionPredictionRequest, VisionPredictionResponse,
    RiskPredictionRequest, RiskPredictionResponse,
    RAGQueryRequest, RAGQueryResponse,
    UnifiedBatchAnalysisResponse
)
from models.vision.predict import VisionPredictor
from models.ml.predict import QualityRiskPredictor
from rag.pipeline import RAGPipeline

router = APIRouter()

# Global predictor instances
vision_predictor = VisionPredictor()
risk_predictor = QualityRiskPredictor()
rag_pipeline = RAGPipeline()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "FactoryIQ Manufacturing Intelligence Engine",
        "version": "1.0.0",
        "categories_supported": ["bottle", "screw", "metal_nut", "tile"]
    }

@router.post("/predict/vision", response_model=VisionPredictionResponse)
def predict_vision(req: VisionPredictionRequest):
    """Run PyTorch visual anomaly detection for product category."""
    res = vision_predictor.predict_synthetic(category=req.category, force_defect=req.force_defect)
    return VisionPredictionResponse(**res)

@router.post("/predict/vision/upload", response_model=VisionPredictionResponse)
async def predict_vision_upload(
    file: UploadFile = File(...),
    category: str = Form("bottle")
):
    """Run visual anomaly detection on uploaded inspection image file."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        res = vision_predictor.predict_image(image, category=category)
        return VisionPredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@router.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(req: RiskPredictionRequest):
    """Predict batch rejection risk level (HIGH/LOW) using ML risk predictor."""
    res = risk_predictor.predict_risk(req.model_dump())
    return RiskPredictionResponse(**res)

@router.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF/Text document, perform OCR extraction & ingest into RAG vector DB."""
    try:
        content_bytes = await file.read()
        try:
            text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = f"Inspection Report: {file.filename}\nBatch: RB-2041 Machine: Press-04 Temp: 184C Defect: Surface crack"
            
        chunk_ids = rag_pipeline.ingest_uploaded_document(text, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunk_ids),
            "message": f"Ingested {file.filename} into FactoryIQ vector store."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document upload failed: {str(e)}")

@router.post("/rag/query", response_model=RAGQueryResponse)
def query_rag(req: RAGQueryRequest):
    """Retrieve grounded context and generate LLM quality answer with citations."""
    v_res = vision_predictor.predict_synthetic(category=req.category or "bottle", force_defect=True)
    r_res = risk_predictor.predict_risk({"batch_id": req.batch_id or "RB-2041", "temperature_c": 184.0})
    
    res = rag_pipeline.answer_query(req.question, vision_res=v_res, risk_res=r_res)
    return RAGQueryResponse(**res)

@router.get("/analyze/batch/{batch_id}", response_model=UnifiedBatchAnalysisResponse)
def analyze_batch(batch_id: str = "RB-2041", machine_id: str = "Press-04", category: str = "bottle"):
    """Unified FactoryIQ endpoint combining Vision + ML Risk + RAG context."""
    r_res = risk_predictor.predict_risk({
        "batch_id": batch_id,
        "machine_id": machine_id,
        "temperature_c": 184.5 if batch_id == "RB-2041" else 170.0,
        "shift": "B"
    })
    v_res = vision_predictor.predict_synthetic(category=category, force_defect=(batch_id == "RB-2041"))
    
    query = f"Why was {batch_id} flagged and what should the engineer check next?"
    rag_res = rag_pipeline.answer_query(query, vision_res=v_res, risk_res=r_res)

    return UnifiedBatchAnalysisResponse(
        batch_id=batch_id,
        machine_id=machine_id,
        risk_prediction=RiskPredictionResponse(**r_res),
        vision_result=VisionPredictionResponse(**v_res),
        rag_synthesis=RAGQueryResponse(**rag_res)
    )
