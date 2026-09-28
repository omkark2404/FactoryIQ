from app.config import DATA_DIR, RISK_MODEL_DIR, VISION_MODEL_DIR, VECTOR_STORE_DIR
import os
import json
import numpy as np
from rag.chunker import TextChunk
from rag.embeddings import EmbeddingGenerator

class VectorStore:
    """Vector Database storing embedded text chunks with metadata search capabilities."""
    def __init__(self, persistence_dir: str = str(VECTOR_STORE_DIR)):
        self.persistence_dir = persistence_dir
        self.embedding_gen = EmbeddingGenerator()
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)

    def add_chunks(self, chunks: list[TextChunk]):
        """Adds text chunks, generates vector embeddings, and updates store index."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vectors = self.embedding_gen.embed_texts(texts)

        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.chunks.extend(chunks)
        print(f"[VectorStore] Indexed {len(chunks)} new chunks. Total indexed: {len(self.chunks)}")

    def similarity_search(self, query: str, top_k: int = 4, filter_meta: dict = None) -> list[tuple[TextChunk, float]]:
        """
        Cosine similarity vector search.
        Returns list of tuples: (TextChunk, similarity_score)
        """
        if len(self.chunks) == 0:
            return []

        query_vec = self.embedding_gen.embed_query(query)
        scores = np.dot(self.vectors, query_vec)  # Cosine similarity for unit-normalized vectors

        # Sort indices descending by similarity score
        sorted_indices = np.argsort(scores)[::-1]

        results = []
        for idx in sorted_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]

            # Apply metadata filtering if specified
            if filter_meta:
                match = True
                for k, v in filter_meta.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def save_index(self):
        """Persists vector store metadata and chunk texts to disk."""
        os.makedirs(self.persistence_dir, exist_ok=True)
        store_data = []
        for chunk in self.chunks:
            store_data.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "source_doc_id": chunk.source_doc_id
            })
        
        json_path = os.path.join(self.persistence_dir, "chunks.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(store_data, f, indent=2)

        np_path = os.path.join(self.persistence_dir, "vectors.npy")
        np.save(np_path, self.vectors)
        print(f"[VectorStore] Persisted index to {self.persistence_dir}")

    def load_index(self):
        """Loads vector store index from disk if available."""
        json_path = os.path.join(self.persistence_dir, "chunks.json")
        np_path = os.path.join(self.persistence_dir, "vectors.npy")

        if os.path.exists(json_path) and os.path.exists(np_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chunks = [TextChunk(**item) for item in data]
            self.vectors = np.load(np_path)
            print(f"[VectorStore] Loaded {len(self.chunks)} chunks from persisted index.")
            return True
        return False
