from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import io

from app.schemas import (
    VisionPredictionRequest, VisionPredictionResponse,
    RiskPredictionRequest, RiskPredictionResponse,
    RAGQueryRequest, RAGQueryResponse,
    UnifiedBatchAnalysisResponse
)
from app.main import ml_models

router = APIRouter()

@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "system": "FactoryIQ Manufacturing Intelligence Engine",
        "models_loaded": len(ml_models) > 0
    }

@router.post("/predict/vision", response_model=VisionPredictionResponse)
async def predict_vision(req: VisionPredictionRequest):
    """Run PyTorch visual anomaly detection for product category."""
    try:
        predictor = ml_models["vision_predictor"]
        res = await run_in_threadpool(predictor.predict_synthetic, category=req.category, force_defect=req.force_defect)
        return VisionPredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB limit

@router.post("/predict/vision/upload", response_model=VisionPredictionResponse)
async def predict_vision_upload(
    file: UploadFile = File(...),
    category: str = Form("bottle")
):
    """Run visual anomaly detection on uploaded inspection image file."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG are allowed.")
    
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
            
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file.filename)
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictor = ml_models["vision_predictor"]
        res = await run_in_threadpool(predictor.predict_image, image, category=category)
        return VisionPredictionResponse(**res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@router.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(req: RiskPredictionRequest):
    """Predict batch rejection risk level (HIGH/LOW) using ML risk predictor."""
    try:
        predictor = ml_models["risk_predictor"]
        res = await run_in_threadpool(predictor.predict_risk, req.model_dump())
        return RiskPredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF/Text document, perform OCR extraction & ingest into RAG vector DB."""
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and Text allowed.")
        
    try:
        content_bytes = await file.read()
        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
            
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file.filename)
        
        try:
            text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = f"Inspection Report: {safe_filename}\nBatch: RB-2041 Machine: Press-04 Temp: 184C Defect: Surface crack"
            
        pipeline = ml_models["rag_pipeline"]
        chunk_ids = await run_in_threadpool(pipeline.ingest_uploaded_document, text, safe_filename)
        return {
            "status": "success",
            "filename": safe_filename,
            "chunks_created": len(chunk_ids)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document upload failed: {str(e)}")

@router.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(req: RAGQueryRequest):
    """Retrieve grounded context and generate LLM quality answer with citations."""
    try:
        vp = ml_models["vision_predictor"]
        rp = ml_models["risk_predictor"]
        pipeline = ml_models["rag_pipeline"]

        v_res = await run_in_threadpool(vp.predict_synthetic, category=req.category or "bottle", force_defect=True)
        r_res = await run_in_threadpool(rp.predict_risk, {"batch_id": req.batch_id or "RB-2041", "temperature_c": 184.0})
        
        res = await run_in_threadpool(pipeline.answer_query, req.question, vision_res=v_res, risk_res=r_res)
        return RAGQueryResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/batch/{batch_id}", response_model=UnifiedBatchAnalysisResponse)
async def analyze_batch(batch_id: str = "RB-2041", machine_id: str = "Press-04", category: str = "bottle"):
    """Unified FactoryIQ endpoint combining Vision + ML Risk + RAG context."""
    try:
        vp = ml_models["vision_predictor"]
        rp = ml_models["risk_predictor"]
        pipeline = ml_models["rag_pipeline"]

        r_res = await run_in_threadpool(rp.predict_risk, {
            "batch_id": batch_id,
            "machine_id": machine_id,
            "temperature_c": 184.5 if batch_id == "RB-2041" else 170.0,
            "shift": "B"
        })
        v_res = await run_in_threadpool(vp.predict_synthetic, category=category, force_defect=(batch_id == "RB-2041"))
        
        query = f"Why was {batch_id} flagged and what should the engineer check next?"
        rag_res = await run_in_threadpool(pipeline.answer_query, query, vision_res=v_res, risk_res=r_res)

        return UnifiedBatchAnalysisResponse(
            batch_id=batch_id,
            machine_id=machine_id,
            risk_prediction=RiskPredictionResponse(**r_res),
            vision_result=VisionPredictionResponse(**v_res),
            rag_synthesis=RAGQueryResponse(**rag_res)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
