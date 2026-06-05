from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


class PDFParser:
    def parse_pages(self, path: str) -> List[Dict[str, Any]]:
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
