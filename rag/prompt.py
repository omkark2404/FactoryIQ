class RAGPromptFormatter:
    """Formats grounding context (CV + ML Risk + Retrieved RAG documents) into LLM system prompts."""
    
    @staticmethod
    def build_quality_assistant_prompt(
        user_query: str,
        vision_res: dict,
        risk_res: dict,
        retrieved_chunks: list[dict]
    ) -> str:
        """Constructs prompt for multi-source quality intelligence synthesis."""
        
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_str += f"--- Source Document [{i}]: {chunk['badge_title']} ({chunk['source_file']}) ---\n"
            context_str += f"{chunk['text']}\n\n"

        prompt = f"""You are FactoryIQ, an expert AI Manufacturing Quality Engineering Assistant.

USER QUESTION:
"{user_query}"

LIVE BATCH DIAGNOSTICS & EVIDENCE:
1. Computer Vision Inspection Result:
   - Status: {vision_res.get('result', 'ANOMALY')}
   - Anomaly Score: {vision_res.get('anomaly_score', 0.91)} (Confidence: {vision_res.get('confidence_pct', 91.0)}%)
   - Detected Defect Region: {vision_res.get('detected_region', 'Surface defect')}

2. Machine Learning Rejection Risk Prediction:
   - Rejection Risk Level: {risk_res.get('risk_level', 'HIGH')}
   - Rejection Probability: {risk_res.get('rejection_risk_pct', 78.0)}%
   - Predicted Defect Rate: {risk_res.get('predicted_defect_rate_pct', 2.52)}%

3. Extracted Operating Parameters:
   - Batch ID: {risk_res.get('batch_id', 'RB-2041')}
   - Machine ID: {risk_res.get('machine_id', 'Press-04')}
   - Operating Temperature: {risk_res.get('temperature_c', 184.0)}°C
   - Shift: {risk_res.get('shift', 'B')}

4. Retrieved Documentation & Historical Incident Evidence:
{context_str if context_str else "No additional documentation retrieved."}

INSTRUCTIONS FOR ANSWER:
- Synthesize all evidence concisely into a direct quality engineer recommendation.
- Explicitly explain WHY the batch was flagged, referencing recorded temperatures, SOP limits, visual inspection results, and past incident precedents.
- List specific recommended corrective actions (e.g. temperature control checks, sample inspections).
"""
        return prompt
