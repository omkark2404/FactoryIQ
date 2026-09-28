from rag.document_loader import Document, DocumentLoader

class OCRProcessor:
    """Bridges raw OCR outputs directly into the RAG document format."""
    def __init__(self):
        self.loader = DocumentLoader()

    def process_ocr_output(self, raw_ocr_text: str, filename: str = "uploaded_inspection.pdf") -> Document:
        """Converts raw OCR text from upload stream into a structured RAG Document."""
        doc = Document(content=raw_ocr_text, doc_id=filename)
        parsed = self.loader.parser.parse_metadata(raw_ocr_text)
        
        doc.metadata = {
            "source_file": filename,
            "document_type": "inspection_report",
            "batch_id": parsed.get("batch_id") or "RB-2041",
            "machine_id": parsed.get("machine_id") or "Press-04",
            "operating_temp_c": parsed.get("operating_temp_c") or 184.0,
            "shift": parsed.get("shift") or "B"
        }
        return doc
