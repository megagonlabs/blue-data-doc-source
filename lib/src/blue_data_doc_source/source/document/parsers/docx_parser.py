from pathlib import Path
from typing import List

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph


_HEADING_STYLES = {
    "Title":     1,
    "Subtitle":  2,
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Heading 6": 6,
}


def _table_to_lines(table: DocxTable) -> List[str]:
    """Render a table as pipe-delimited rows so chunkers treat it as readable text."""
    lines = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


class DocxParser:
    """Extracts text from .docx files, converting headings to Markdown-style headers.

    Heading styles (Title, Heading 1–6) become ``# / ## / ...`` prefixes so
    ``SectionChunker`` can detect section boundaries without any extra config.
    Tables are rendered as pipe-delimited rows. Empty paragraphs are skipped.
    Document order (paragraphs and tables interleaved) is preserved.
    """

    def parse(self, path: str) -> str:
        """Return the full document text with heading structure preserved.

        Args:
            path: Absolute or relative path to a ``.docx`` file.

        Returns:
            Full text as a single string. Headings are prefixed with ``#``
            markers; table rows are pipe-delimited.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not ``.docx``.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                f"DocxParser only supports .docx files, got: {file_path.suffix}"
            )

        doc = Document(str(file_path))
        lines: List[str] = []

        for child in doc.element.body:
            if child.tag == qn("w:p"):
                para = DocxParagraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name if para.style else "Normal"
                level = _HEADING_STYLES.get(style_name)
                if level:
                    lines.append("#" * level + " " + text)
                else:
                    lines.append(text)

            elif child.tag == qn("w:tbl"):
                table = DocxTable(child, doc)
                lines.extend(_table_to_lines(table))

        return "\n\n".join(lines)
