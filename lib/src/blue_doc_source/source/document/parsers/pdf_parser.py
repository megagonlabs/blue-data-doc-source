from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


class PDFParser:
    """Extracts text from a PDF file one page at a time using pypdf."""

    def parse_pages(self, path: str) -> List[Dict[str, Any]]:
        """Return per-page text from a PDF file, skipping blank pages.

        Args:
            path: Absolute or relative path to a .pdf file.

        Returns:
            List of dicts with keys ``page_num`` (1-based int) and ``text`` (str).
            Pages with no extractable text are omitted.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pdf.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDFParser only supports .pdf files, got: {file_path.suffix}")

        reader = PdfReader(str(file_path))
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_num": page_num, "text": text})

        return pages
