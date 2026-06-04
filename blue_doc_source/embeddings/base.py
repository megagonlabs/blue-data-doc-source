from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError