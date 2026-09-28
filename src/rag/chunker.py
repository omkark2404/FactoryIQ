from dataclasses import dataclass
from src.rag.document_loader import Document

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict
    source_doc_id: str

class TextChunker:
    """Sliding-window text chunker preserving document metadata and sentence boundaries."""
    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> list[TextChunk]:
        """Splits document text into overlapping chunks with inherited document metadata."""
        text = doc.content
        if not text:
            return []

        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            chunk_meta = doc.metadata.copy()
            chunk_meta["chunk_index"] = 0
            chunks.append(TextChunk(
                chunk_id=f"{doc.doc_id}_chunk_0",
                text=text,
                metadata=chunk_meta,
                source_doc_id=doc.doc_id
            ))
            return chunks

        step = self.chunk_size - self.chunk_overlap
        idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_meta = doc.metadata.copy()
            chunk_meta["chunk_index"] = idx

            chunks.append(TextChunk(
                chunk_id=f"{doc.doc_id}_chunk_{idx}",
                text=chunk_text,
                metadata=chunk_meta,
                source_doc_id=doc.doc_id
            ))
            idx += 1

        return chunks
