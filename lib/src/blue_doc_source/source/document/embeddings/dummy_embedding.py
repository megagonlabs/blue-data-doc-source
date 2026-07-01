import hashlib
from typing import List

from .base import EmbeddingModel


class DummyEmbeddingModel(EmbeddingModel):
    """Deterministic embedding model for tests — not suitable for real similarity search.

    Derives a fixed-length float vector from the SHA-256 hash of the input text.
    Vectors are in [0, 1] but carry no semantic meaning.
    """

    def __init__(self, dim: int = 8):
        """
        Args:
            dim: Output vector length (max 32, limited by SHA-256 digest size).
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        values = []
        for i in range(self.dim):
            values.append(digest[i] / 255.0)

        return values