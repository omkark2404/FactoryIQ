import os
from dataclasses import dataclass, field
from models.ocr.extract import OCRExtractor
from models.ocr.clean import OCRTextCleaner
from models.ocr.parser import OCRDocumentParser

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

class DocumentLoader:
    """Document Loader for SOPs, Inspection Reports, Quality Manuals & Incident Logs."""
    def __init__(self):
        self.extractor = OCRExtractor()
        self.cleaner = OCRTextCleaner()
        self.parser = OCRDocumentParser()

    def load_file(self, file_path: str, doc_type: str = "general") -> Document:
        """Loads and parses a single file into a structured Document with metadata."""
        raw_text = self.extractor.extract_text_from_file(file_path)
        clean_text = self.cleaner.clean_text(raw_text)
        parsed_meta = self.parser.parse_metadata(clean_text)

        fname = os.path.basename(file_path)
        doc_id = os.path.splitext(fname)[0]

        # Basic doc_type inference if not provided
        if "sop" in fname.lower():
            doc_type = "machine_sop"
        elif "inspection" in fname.lower():
            doc_type = "inspection_report"
        elif "incident" in fname.lower():
            doc_type = "incident_report"
        elif "manual" in fname.lower():
            doc_type = "quality_manual"

        metadata = {
            "source_file": fname,
            "document_type": doc_type,
            "batch_id": parsed_meta.get("batch_id"),
            "machine_id": parsed_meta.get("machine_id"),
            "operating_temp_c": parsed_meta.get("operating_temp_c"),
            "shift": parsed_meta.get("shift")
        }

        return Document(content=clean_text, metadata=metadata, doc_id=doc_id)

    def load_directory(self, dir_path: str) -> list[Document]:
        """Loads all supported documents from directory."""
        docs = []
        if not os.path.exists(dir_path):
            return docs
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.txt', '.pdf', '.log', '.png', '.jpg')):
                    full_path = os.path.join(root, file)
                    docs.append(self.load_file(full_path))
        return docs
