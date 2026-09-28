from src.ocr.clean import OCRTextCleaner

class RAGTextCleaner(OCRTextCleaner):
    """Clean document text specifically tailored for RAG chunking."""
    
    @classmethod
    def clean_for_rag(cls, text: str) -> str:
        cleaned = cls.clean_text(text)
        # Strip header noise lines like page numbers
        lines = cleaned.split('\n')
        filtered = [line for line in lines if not line.strip().isdigit()]
        return '\n'.join(filtered)
