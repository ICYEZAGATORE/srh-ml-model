"""
embedder.py — Reusable embedding module for the SRH RAG pipeline.
Imported by the FastAPI backend (srh-backend-api) at inference time.
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class SRHEmbedder:
    """
    Wraps the multilingual sentence-transformer for SRH content.
    Supports English and Kinyarwanda queries.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts:     List of strings to embed.
            normalize: L2-normalise for cosine similarity via dot product.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed([query])


# Module-level singleton (lazy-loaded on first import)
_embedder: SRHEmbedder | None = None

def get_embedder() -> SRHEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SRHEmbedder()
    return _embedder
