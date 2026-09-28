import numpy as np
from typing import List

class EmbeddingGenerator:
    """
    Dense embedding generator wrapper.
    Uses SentenceTransformers ('all-MiniLM-L6-v2') if installed/online,
    or a normalized feature hashing fallback vector encoder for local offline robustness.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st_model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name)
            print(f"[EmbeddingGenerator] Loaded SentenceTransformer model '{model_name}'")
        except Exception:
            print(f"[EmbeddingGenerator] Notice: SentenceTransformers fallback to local vector encoder.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into normalized vectors (numpy ndarray)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return (embeddings / norms).astype(np.float32)

        # Robust hashing fallback encoder (384-dimensional vector space)
        return self._hash_vector_encoder(texts, dim=384)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query string."""
        return self.embed_texts([query])[0]

    def _hash_vector_encoder(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """Deterministic TF-IDF hash fallback embedding generator."""
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                h = abs(hash(word)) % dim
                matrix[row, h] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix
