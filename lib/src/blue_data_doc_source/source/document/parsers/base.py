from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Parser(ABC):
    """Abstract base class for document parsers.

    Implement this to bring in an external document processor — LlamaIndex,
    a vision model, a cloud OCR API, etc. — and pass the instance as
    ``parser=`` to ``DocumentPipeline``. The pipeline will call ``parse()``
    instead of its built-in ``TextParser`` / ``PDFParser``.

    ``parse()`` must return a list of page dicts. Each dict requires a
    ``"text"`` key (str). ``"page_num"`` (int, 1-based) is optional — include
    it for page-aware documents so provenance fields are populated on each
    ``ChunkRecord``.

    Example::

        class LlamaIndexParser(Parser):
            def parse(self, path: str) -> List[Dict[str, Any]]:
                from llama_index.core import SimpleDirectoryReader
                docs = SimpleDirectoryReader(input_files=[path]).load_data()
                return [{"text": doc.text} for doc in docs]

        pipeline = DocumentPipeline(parser=LlamaIndexParser(), embedding_model=...)
        pipeline.ingest("complex_layout.pdf")
    """

    @abstractmethod
    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Parse a document file and return its content as a list of page dicts.

        Args:
            path: Path to the document file.

        Returns:
            List of dicts, each with at minimum ``{"text": str}``.
            Include ``"page_num": int`` for page-aware documents.
        """
        raise NotImplementedError  # pragma: no cover

    def parse_blocks(self, path: str) -> List[Dict[str, Any]]:
        """Parse a document into typed content blocks.

        Returns a flat list of block dicts. Each block has at minimum
        ``{"type": str, "text": str}``. Supported types:

        - ``"text"`` — prose or narrative content
        - ``"table"`` — structured tabular data; also carries a ``"table"``
          key with ``{"headers": list[str], "rows": list[list[str]]}``

        ``"page_num"`` (int, 1-based) is included when available.

        The default implementation wraps ``parse()`` and marks every block
        as ``type="text"``, so implementing only ``parse()`` is sufficient
        for parsers that return flat text (e.g. a vision model that extracts
        plain text, or a LlamaIndex wrapper). Override ``parse_blocks()``
        when the parser can distinguish content types — e.g. PlumberPDFParser
        for tables, Docling for rich layouts, or a vision model prompted to
        return structured JSON with table and prose blocks separated.
        """
        return [{"type": "text", **page} for page in self.parse(path)]
