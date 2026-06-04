import hashlib
from typing import List

from .base import EmbeddingModel


class DummyEmbeddingModel(EmbeddingModel):
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        values = []
        for i in range(self.dim):
            values.append(digest[i] / 255.0)

        return values