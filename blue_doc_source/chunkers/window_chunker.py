from typing import List


class WindowChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
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