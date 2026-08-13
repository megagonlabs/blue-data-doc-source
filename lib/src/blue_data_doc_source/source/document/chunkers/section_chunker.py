import re
from typing import List, Optional, Tuple

from .base import Chunker
from .window_chunker import WindowChunker

# Markdown headers: # Title, ## Section, ### Sub
_MD_HEADER = re.compile(r'^#{1,6}\s+(.+)$')
# Numbered sections: 1. Title, 2.3 Overview, 4.1.2 Detail
_NUM_HEADER = re.compile(r'^\d+(?:\.\d+)*[.)]\s+\S')
# ALL-CAPS headings (3–8 words, no lowercase)
_CAPS_HEADER = re.compile(r'^[A-Z][A-Z0-9 \t\-]{3,}$')
# Underline markers (=== or ---)
_UNDERLINE = re.compile(r'^[=\-]{3,}\s*$')


def _extract_title(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    m = _MD_HEADER.match(stripped)
    if m:
        return m.group(1).strip()
    if _NUM_HEADER.match(stripped):
        return stripped
    if _CAPS_HEADER.match(stripped) and len(stripped.split()) <= 8:
        return stripped
    return None


class SectionChunker(Chunker):
    """Splits text by section headers, then applies WindowChunker within each section.

    Detects markdown headers (``# Title``), numbered sections (``1.2 Overview``),
    ALL-CAPS headings, and underlined headings (``Title\\n======``). When no
    headers are found, falls back to plain window chunking with ``section_title=None``.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        """
        Args:
            chunk_size: Maximum words per chunk passed to the inner WindowChunker.
            overlap: Word overlap between consecutive chunks.
        """
        self._window = WindowChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        sections = self._split_sections(text)
        if not sections:
            return [(None, chunk) for chunk in self._window.chunk(text)]

        result = []
        for title, content in sections:
            content = content.strip()
            if not content:
                continue
            for chunk in self._window.chunk(content):
                result.append((title, chunk))
        return result

    def _split_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        lines = text.splitlines()
        sections: List[Tuple[Optional[str], str]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Underlined header: line followed by === or ---
            if i + 1 < len(lines) and _UNDERLINE.match(lines[i + 1]):
                candidate = line.strip()
                if candidate:
                    sections.append((current_title, "\n".join(current_lines)))
                    current_title = candidate
                    current_lines = []
                    i += 2
                    continue

            title = _extract_title(line)
            if title is not None:
                sections.append((current_title, "\n".join(current_lines)))
                current_title = title
                current_lines = []
            else:
                current_lines.append(line)

            i += 1

        sections.append((current_title, "\n".join(current_lines)))

        has_named = any(t is not None for t, _ in sections)
        return sections if has_named else []
