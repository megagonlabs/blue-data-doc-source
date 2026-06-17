from abc import ABC, abstractmethod
from typing import Any, Dict


class Enricher(ABC):
    """Abstract base class for chunk enrichment."""

    @abstractmethod
    def enrich(self, text: str) -> Dict[str, Any]:
        """Extract structured metadata from a chunk of text.

        Args:
            text: The chunk text to analyse.

        Returns:
            A dict with two keys:
            - ``entities``: list of ``{text: str, type: str}`` dicts
            - ``semantic_hints``: dict of scalar metadata about the chunk
        """
        raise NotImplementedError
