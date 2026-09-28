from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class VisionPredictionRequest(BaseModel):
    category: str = Field("bottle", description="Product category: bottle, screw, metal_nut, tile")
    force_defect: bool = Field(True, description="For testing: force defect anomaly in synthetic mode")

class VisionPredictionResponse(BaseModel):
    category: str
    result: str  # ANOMALY or NORMAL
    anomaly_score: float
    confidence_pct: float
    detected_region: str
    status_message: str

class RiskPredictionRequest(BaseModel):
    batch_id: str = "RB-2041"
    machine_id: str = "Press-04"
    temperature_c: float = 184.0
    pressure_psi: float = 72.1
    line_speed_mmin: float = 45.0
    shift: str = "B"
    previous_defects: int = 4

class RiskPredictionResponse(BaseModel):
    batch_id: str
    machine_id: str
    shift: str
    temperature_c: float
    risk_level: str  # HIGH or LOW
    rejection_risk_pct: float
    predicted_defect_rate_pct: float
    recommendation: str

class RAGQueryRequest(BaseModel):
    question: str = "Why was RB-2041 flagged?"
    category: Optional[str] = "bottle"
    batch_id: Optional[str] = "RB-2041"

class SourceCitation(BaseModel):
    title: str
    file: str
    score: float

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunks: List[Dict[str, Any]]

class UnifiedBatchAnalysisResponse(BaseModel):
    batch_id: str
    machine_id: str
    risk_prediction: RiskPredictionResponse
    vision_result: VisionPredictionResponse
    rag_synthesis: RAGQueryResponse
