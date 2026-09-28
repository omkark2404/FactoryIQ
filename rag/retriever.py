from rag.vector_store import VectorStore
from rag.chunker import TextChunk

class RAGRetriever:
    """Retriever engine fetching contextually relevant document chunks for user queries."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4, filter_meta: dict = None) -> list[dict]:
        """
        Retrieves top-k relevant document chunks with formatted metadata and source tags.
        """
        results = self.vector_store.similarity_search(query, top_k=top_k, filter_meta=filter_meta)
        retrieved = []
        for chunk, score in results:
            source_file = chunk.metadata.get("source_file", chunk.source_doc_id)
            doc_type = chunk.metadata.get("document_type", "Document")
            
            # Form clean user-facing citation tag
            badge_title = f"{doc_type.replace('_', ' ').title()}"
            if chunk.metadata.get("batch_id"):
                badge_title += f" {chunk.metadata['batch_id']}"
            elif chunk.metadata.get("machine_id"):
                badge_title += f" ({chunk.metadata['machine_id']})"
                
            retrieved.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": round(float(score), 4),
                "source_file": source_file,
                "badge_title": badge_title,
                "metadata": chunk.metadata
            })
        return retrieved
