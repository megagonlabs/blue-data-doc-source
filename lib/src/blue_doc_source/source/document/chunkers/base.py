from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        """Split text into chunks, returning (section_title, chunk_text) pairs.

        Args:
            text: Input text to split.

        Returns:
            List of (section_title, chunk_text) tuples. ``section_title`` is
            ``None`` when the chunk does not belong to a named section.
        """
        raise NotImplementedError
