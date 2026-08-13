from typing import List, Optional, Tuple

from .base import Chunker


class WindowChunker(Chunker):
    """Splits text into overlapping chunks using a word-count sliding window.

    Chunk boundaries are word-aligned, so chunk sizes are exact in words but
    variable in characters. Overlap preserves context across chunk boundaries.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        """
        Args:
            chunk_size: Maximum number of words per chunk.
            overlap: Number of words shared between consecutive chunks.

        Raises:
            ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping word-window chunks.

        Args:
            text: Input text to chunk. Whitespace is used as the word delimiter.

        Returns:
            List of chunk strings. Returns an empty list if text is blank.
        """
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(chunk_text)

            start = end - self.overlap

        return chunks

    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        return [(None, chunk) for chunk in self.chunk(text)]