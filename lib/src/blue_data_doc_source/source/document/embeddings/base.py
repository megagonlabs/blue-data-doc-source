from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for text embedding models."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimension. Must match the target vector index configuration."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a string and return a fixed-length float vector.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError  # pragma: no cover