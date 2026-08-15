from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Parser


def _to_pipe_delimited(headers: List[str], rows: List[List[str]]) -> str:
    """Render headers + rows as a markdown-style pipe-delimited table."""
    def _row(cells):
        return "| " + " | ".join(cells) + " |"

    lines = []
    if headers:
        lines.append(_row(headers))
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append(_row(row))
    return "\n".join(lines)


def _clean_cell(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _in_bbox(obj: Dict[str, Any], bbox: tuple, tolerance: float = 2.0) -> bool:
    """Return True when obj falls inside bbox (x0, top, x1, bottom)."""
    x0, top, x1, bottom = bbox
    return (
        obj.get("x0", 0) >= x0 - tolerance
        and obj.get("x1", 0) <= x1 + tolerance
        and obj.get("top", 0) >= top - tolerance
        and obj.get("bottom", 0) <= bottom + tolerance
    )


class PlumberPDFParser(Parser):
    """PDF parser that extracts tables as structured blocks alongside prose text.

    Uses ``pdfplumber`` to detect table boundaries on each page. Tables are
    returned as ``content_type="table"`` blocks carrying both a
    pipe-delimited ``text`` (for LLM consumption) and a structured ``table``
    dict (``headers`` + ``rows``) for machine-readable use. Prose text
    outside table regions is returned as ``content_type="text"`` blocks.

    Requires ``pdfplumber``::

        pip install pdfplumber

    Usage::

        from blue_data_doc_source import DocumentPipeline, PlumberPDFParser

        pipeline = DocumentPipeline(
            parser=PlumberPDFParser(),
            embedding_model=...,
            vector_source=...,
        )
        records = pipeline.ingest("invoice.pdf")
        # Table chunks carry content_type="table" and a structured table field
        tables = [r for r in records if r.get("content_type") == "table"]
    """

    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Merge all blocks per page into single page dicts for backward compat."""
        blocks = self.parse_blocks(path)
        pages: Dict[Optional[int], List[str]] = {}
        order: List[Optional[int]] = []
        for block in blocks:
            pn = block.get("page_num")
            if pn not in pages:
                pages[pn] = []
                order.append(pn)
            pages[pn].append(block["text"])
        return [
            {"page_num": pn, "text": "\n\n".join(texts)} if pn is not None
            else {"text": "\n\n".join(texts)}
            for pn in order
            for texts in [pages[pn]]
        ]

    def parse_blocks(self, path: str) -> List[Dict[str, Any]]:
        """Return typed content blocks extracted page by page.

        Args:
            path: Path to a .pdf file.

        Returns:
            List of block dicts. Each has ``type``, ``text``, and
            ``page_num``. Table blocks additionally carry ``table``:
            ``{"headers": list[str], "rows": list[list[str]]}``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pdf.
            ImportError: If pdfplumber is not installed.
        """
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PlumberPDFParser. "
                "Run: pip install pdfplumber"
            ) from exc

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"PlumberPDFParser only supports .pdf files, got: {file_path.suffix}")

        blocks: List[Dict[str, Any]] = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                blocks.extend(self._blocks_from_page(page, page_num))
        return blocks

    def _blocks_from_page(self, page: Any, page_num: int) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        found_tables = page.find_tables()

        if not found_tables:
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append({"type": "text", "page_num": page_num, "text": text})
            return blocks

        table_bboxes = [t.bbox for t in found_tables]

        # Text outside all table regions
        def outside_tables(obj):
            return not any(_in_bbox(obj, bbox) for bbox in table_bboxes)

        text = (page.filter(outside_tables).extract_text() or "").strip()
        if text:
            blocks.append({"type": "text", "page_num": page_num, "text": text})

        # One block per table, in page order (top-to-bottom)
        table_entries = []
        for table_obj in found_tables:
            rows = table_obj.extract()
            if not rows:
                continue
            cleaned = [[_clean_cell(c) for c in row] for row in rows]
            if not any(any(c for c in row) for row in cleaned):
                continue  # entirely empty table

            headers = cleaned[0]
            data_rows = cleaned[1:]
            pipe_text = _to_pipe_delimited(headers, data_rows)
            table_entries.append((
                table_obj.bbox[1],  # top y — for ordering
                {
                    "type": "table",
                    "page_num": page_num,
                    "text": pipe_text,
                    "table": {"headers": headers, "rows": data_rows},
                }
            ))

        # Sort tables top-to-bottom so reading order is preserved
        for _, block in sorted(table_entries, key=lambda x: x[0]):
            blocks.append(block)

        return blocks
