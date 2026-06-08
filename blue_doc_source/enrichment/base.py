from abc import ABC, abstractmethod
from typing import Any, Dict


class Enricher(ABC):
    @abstractmethod
    def enrich(self, text: str) -> Dict[str, Any]:
        """Return {"entities": [...], "semantic_hints": {...}}."""
        raise NotImplementedError
