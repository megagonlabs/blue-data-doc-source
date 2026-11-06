from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


def _preprocess_image(img: Any) -> Any:
    """Grayscale + contrast boost + binarise to improve Tesseract accuracy on scanned pages."""
    from PIL import ImageEnhance
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.point(lambda x: 0 if x < 128 else 255)
    return img


class PDFParser:
    """Extracts text from a PDF file one page at a time using pypdf.

    For scanned PDFs (pages with no text layer), pass ``ocr_fallback=True``
    to run Tesseract OCR on blank pages. Requires the ``[ocr]`` extras:
    ``pip install -e "lib/.[ocr]"`` (from the repo root; see docs/installation.md).
    """

    def __init__(self, ocr_fallback: bool = False):
        """
        Args:
            ocr_fallback: When ``True``, pages with no extractable text are
                converted to images, pre-processed, and run through Tesseract
                OCR. Each OCR'd page includes an ``ocr_confidence`` score
                (0–100). Requires ``pytesseract`` and ``pdf2image``
                (install with ``[ocr]`` extras).
        """
        self._ocr_fallback = ocr_fallback

    def parse_pages(self, path: str) -> List[Dict[str, Any]]:
        """Return per-page text from a PDF file, skipping blank pages.

        Args:
            path: Absolute or relative path to a .pdf file.

        Returns:
            List of dicts with keys ``page_num`` (1-based int) and ``text``
            (str). OCR'd pages additionally carry ``"ocr": True`` and
            ``"ocr_confidence"`` (float, mean word confidence 0–100).
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
                ocr_result = self._ocr_page(str(file_path), page_num)
                if ocr_result.get("text"):
                    pages.append({
                        "page_num": page_num,
                        "text": ocr_result["text"],
                        "ocr": True,
                        "ocr_confidence": ocr_result["ocr_confidence"],
                    })

        return pages

    def _ocr_page(self, path: str, page_num: int) -> Dict[str, Any]:
        """OCR a single PDF page and return extracted text with a confidence score.

        Returns:
            Dict with ``"text"`` (str) and ``"ocr_confidence"`` (float, 0–100)
            when text is found. Empty dict when the page yields no output.

        Raises:
            ImportError: If ``pdf2image`` or ``pytesseract`` are not installed.
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:
            raise ImportError(
                "OCR dependencies not installed. "
                'Run (from the repo root): pip install -e "lib/.[ocr]"'
            ) from exc

        images = convert_from_path(path, first_page=page_num, last_page=page_num)
        if not images:
            return {}

        img = _preprocess_image(images[0])
        data = pytesseract.image_to_data(img, output_type=Output.DICT)

        words = []
        confidences = []
        for i in range(len(data["text"])):
            conf = int(data["conf"][i])
            word = data["text"][i].strip()
            if conf > 0 and word:
                words.append(word)
                confidences.append(conf)

        text = " ".join(words).strip()
        if not text:
            return {}

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {"text": text, "ocr_confidence": round(mean_conf, 1)}
