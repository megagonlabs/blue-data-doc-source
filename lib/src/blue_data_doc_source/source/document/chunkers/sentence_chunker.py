import re
from typing import List, Optional, Tuple

from .base import Chunker


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Uses ``nltk.sent_tokenize`` when available. Falls back to a regex splitter
    when nltk is not installed or punkt_tab data has not been downloaded.
    Install the ``[nlp]`` extra and run ``nltk.download('punkt_tab')`` to
    enable the nltk path.
    """
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except (ImportError, LookupError):
        pass

    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


class SentenceChunker(Chunker):
    """Splits text into overlapping chunks aligned to sentence boundaries.

    Accumulates sentences until the next sentence would push the chunk past
    ``chunk_size`` words, then closes the chunk at the last complete sentence.
    This ensures no sentence is split mid-thought across chunk boundaries.

    Overlap is built by carrying trailing sentences from the previous chunk
    until the overlap word budget is filled.

    When no sentence boundaries are detected (e.g. raw tokens without
    punctuation), falls back to word-count splitting so the chunker degrades
    gracefully on unstructured text.

    Uses ``nltk.sent_tokenize`` when available; install the optional ``[nlp]``
    extra and run ``nltk.download('punkt_tab')`` for higher-quality detection.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        """
        Args:
            chunk_size: Target maximum words per chunk. Chunks close at the
                last sentence boundary before this limit is reached.
            overlap: Target words of context carried from the previous chunk.
                Whole sentences are carried until the word budget is filled.

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
        """Split text into sentence-aligned overlapping chunks.

        Args:
            text: Input text to split.

        Returns:
            List of chunk strings. Empty when text is blank.
        """
        if not text:
            return []
        words = text.split()
        if not words:
            return []

        sentences = _split_sentences(text)

        # No sentence boundaries detected — fall back to word-count splitting
        if len(sentences) <= 1:
            return self._word_chunk(words)

        chunks: List[str] = []
        current: List[str] = []
        current_words = 0

        for sent in sentences:
            sent_word_count = len(sent.split())

            if current_words + sent_word_count > self.chunk_size and current:
                chunks.append(" ".join(current))

                # Carry trailing sentences into the next chunk as overlap
                overlap_sents: List[str] = []
                overlap_words = 0
                for s in reversed(current):
                    w = len(s.split())
                    if overlap_words + w <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_words += w
                    else:
                        break
                current = overlap_sents
                current_words = overlap_words

                # Trim carried overlap so the incoming sentence doesn't push the
                # new chunk past chunk_size. #85
                while current and current_words + sent_word_count > self.chunk_size:
                    current_words -= len(current.pop(0).split())

            current.append(sent)
            current_words += sent_word_count

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _word_chunk(self, words: List[str]) -> List[str]:
        """Word-count sliding window — used when no sentence boundaries found."""
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Stop once the window reaches the end, so the overlap step-back
            # doesn't emit a redundant substring tail chunk. #85
            if end >= len(words):
                break
            start = end - self.overlap
        return chunks

    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        """Chunk text with section_title=None on every chunk.

        SentenceChunker has no section awareness — title is always None.
        Use SectionChunker to preserve section headings.
        """
        return [(None, chunk) for chunk in self.chunk(text)]
