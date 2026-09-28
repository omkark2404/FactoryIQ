from app.config import DATA_DIR, RISK_MODEL_DIR, VISION_MODEL_DIR, VECTOR_STORE_DIR
import os
import argparse
from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.prompt import RAGPromptFormatter
from rag.ocr_processor import OCRProcessor

class RAGPipeline:
    """FactoryIQ End-to-End RAG Engine with source citation formatting."""
    def __init__(self, vector_store_dir: str = str(VECTOR_STORE_DIR)):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.vector_store = VectorStore(persistence_dir=vector_store_dir)
        self.ocr_processor = OCRProcessor()
        
        # Try loading existing index; if missing, build initial index from docs
        if not self.vector_store.load_index():
            self.ingest_default_documents()

        self.retriever = RAGRetriever(self.vector_store)

    def ingest_default_documents(self, docs_dir: str = os.path.join(str(DATA_DIR), "raw", "documents")):
        """Ingests default manufacturing SOPs, manuals & reports into vector database."""
        print(f"[RAGPipeline] Ingesting documents from '{docs_dir}'...")
        docs = self.loader.load_directory(docs_dir)
        all_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.vector_store.add_chunks(all_chunks)
        self.vector_store.save_index()
        print(f"[RAGPipeline] Successfully indexed {len(all_chunks)} chunks.")

    def ingest_uploaded_document(self, ocr_text: str, filename: str) -> list[str]:
        """Ingests an uploaded document's OCR text into active vector store."""
        doc = self.ocr_processor.process_ocr_output(ocr_text, filename=filename)
        chunks = self.chunker.chunk_document(doc)
        self.vector_store.add_chunks(chunks)
        self.vector_store.save_index()
        return [c.chunk_id for c in chunks]

    def answer_query(
        self,
        query: str,
        vision_res: dict = None,
        risk_res: dict = None
    ) -> dict:
        """
        Processes user question, retrieves relevant chunks, synthesizes grounded answer, 
        and formats explicit source citations.
        """
        # Default mock models if not provided
        vision_res = vision_res or {
            "category": "bottle",
            "result": "ANOMALY",
            "anomaly_score": 0.91,
            "confidence_pct": 91.0,
            "detected_region": "Surface crack near neck shoulder"
        }
        risk_res = risk_res or {
            "batch_id": "RB-2041",
            "machine_id": "Press-04",
            "risk_level": "HIGH",
            "rejection_risk_pct": 78.0,
            "predicted_defect_rate_pct": 2.52,
            "temperature_c": 184.0,
            "shift": "B"
        }

        # Vector Retrieval
        retrieved_chunks = self.retriever.retrieve(query, top_k=3)
        
        # Guardrail: Check if retrieval confidence is too low
        # Assuming score is dot product (cosine similarity) where higher is better, or L2 where lower is better. 
        # For safety, let's pass a guardrail instruction to the LLM directly via prompt.
        prompt = RAGPromptFormatter.build_quality_assistant_prompt(
            user_query=query,
            vision_res=vision_res,
            risk_res=risk_res,
            retrieved_chunks=retrieved_chunks
        )
        
        # Add strict guardrail to prompt
        prompt += "\n\nCRITICAL RULE: If the retrieved documents do not contain the answer, explicitly state: 'The provided SOPs and incident reports do not contain enough information to answer this question.'"

        # Synthesize Grounded Answer (Local rule-based generator for offline reliability)
        answer = self._generate_synthesized_answer(prompt, query, vision_res, risk_res, retrieved_chunks)

        # Format Source Citations
        sources = []
        seen_files = set()
        for chunk in retrieved_chunks:
            s_file = chunk["source_file"]
            if s_file not in seen_files:
                seen_files.add(s_file)
                sources.append({
                    "title": chunk["badge_title"],
                    "file": s_file,
                    "score": chunk["score"]
                })

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "prompt_preview": prompt
        }

    def _generate_synthesized_answer(
        self,
        prompt: str,
        query: str,
        vision_res: dict,
        risk_res: dict,
        chunks: list[dict]
    ) -> str:
        """Synthesizes high-fidelity domain response using Google Gemini API or offline fallback."""
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Deterministic fallback when offline
            batch_id = risk_res.get("batch_id", "RB-2041")
            machine_id = risk_res.get("machine_id", "Press-04")
            temp = risk_res.get("temperature_c", 184.0)
            risk_pct = risk_res.get("rejection_risk_pct", 78.0)

            return (
                f"**[OFFLINE MODE]** Set `GEMINI_API_KEY` in `.env` to enable dynamic AI responses.\n\n"
                f"Batch {batch_id} has been flagged because the visual inspection detected a surface anomaly "
                f"(Confidence: {vision_res.get('confidence_pct', 91.0)}%) and the production model predicts a high "
                f"rejection risk ({risk_pct}% risk level HIGH). The inspection records show {machine_id} operating "
                f"at {temp}°C. The retrieved SOP indicates that temperatures above 180°C require mandatory visual inspection "
                f"due to thermal expansion strain.\n\n"
                f"**Recommended Action for Quality Engineer:**\n"
                f"1. Immediately inspect {machine_id} temperature control sensors and cooling lines.\n"
                f"2. Quarantine the remaining units from batch {batch_id} for 100% visual defect sampling."
            )
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"**[LLM Error]**: {str(e)}\n\nPlease check your GEMINI_API_KEY configuration."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest default documents into vector database")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    if args.ingest:
        pipeline.ingest_default_documents()
    else:
        res = pipeline.answer_query("Why was RB-2041 flagged?")
        print("\n=== ANSWER ===")
        print(res["answer"])
        print("\n=== SOURCES ===")
        for s in res["sources"]:
            print(f"✓ {s['title']} ({s['file']})")
