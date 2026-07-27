from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


class PDFParser:
    """Extracts text from a PDF file one page at a time using pypdf.

    For scanned PDFs (pages with no text layer), pass ``ocr_fallback=True``
    to run Tesseract OCR on blank pages. Requires the ``[ocr]`` extras:
    ``pip install blue-data-doc-source[ocr]``.
    """

    def __init__(self, ocr_fallback: bool = False):
        """
        Args:
            ocr_fallback: When ``True``, pages with no extractable text are
                converted to images and processed with Tesseract OCR. Requires
                ``pytesseract`` and ``pdf2image`` (install with ``[ocr]`` extras).
        """
        self._ocr_fallback = ocr_fallback

    def parse_pages(self, path: str) -> List[Dict[str, Any]]:
        """Return per-page text from a PDF file, skipping blank pages.

        Args:
            path: Absolute or relative path to a .pdf file.

        Returns:
            List of dicts with keys ``page_num`` (1-based int), ``text`` (str),
            and ``ocr`` (bool, present and ``True`` only when OCR was used).
            Pages with no extractable text (and no OCR output) are omitted.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pdf.
            ImportError: If ``ocr_fallback=True`` but OCR dependencies are missing.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDFParser only supports .pdf files, got: {file_path.suffix}")

        reader = PdfReader(str(file_path))
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append({"page_num": page_num, "text": text})
            elif self._ocr_fallback:
                ocr_text = self._ocr_page(str(file_path), page_num)
                if ocr_text:
                    pages.append({"page_num": page_num, "text": ocr_text, "ocr": True})

        return pages

    def _ocr_page(self, path: str, page_num: int) -> str:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "OCR dependencies not installed. "
                "Run: pip install blue-data-doc-source[ocr]"
            ) from exc

        images = convert_from_path(path, first_page=page_num, last_page=page_num)
        if not images:
            return ""
        return pytesseract.image_to_string(images[0]).strip()
