"""Tests for PlumberPDFParser — block-aware PDF parsing with table extraction."""
from __future__ import annotations

import io
import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel, PlumberPDFParser


# ---------------------------------------------------------------------------
# Helpers — build minimal PDFs in-memory using fpdf
# ---------------------------------------------------------------------------

def _pdf_with_table(headers, rows, prose_before="", prose_after=""):
    """Return bytes of a single-page PDF containing prose + a table."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    if prose_before:
        pdf.multi_cell(0, 6, prose_before)
        pdf.ln(2)

    # Header row
    pdf.set_font("Helvetica", "B", 10)
    col_w = 180 // len(headers)
    for h in headers:
        pdf.cell(col_w, 7, h, border=1)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", size=10)
    for row in rows:
        for cell in row:
            pdf.cell(col_w, 7, cell, border=1)
        pdf.ln()

    if prose_after:
        pdf.ln(2)
        pdf.multi_cell(0, 6, prose_after)

    return pdf.output()


def _pdf_text_only(text):
    """Return bytes of a single-page text-only PDF."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, text)
    return pdf.output()


def _write_pdf(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------------------
# parse_blocks — structural tests
# ---------------------------------------------------------------------------

class TestParseBlocks:
    def test_text_only_pdf_returns_text_blocks(self, tmp_path):
        path = _write_pdf(tmp_path, "a.pdf", _pdf_text_only("Hello world, this is a test document."))
        parser = PlumberPDFParser()
        blocks = parser.parse_blocks(path)
        assert len(blocks) >= 1
        assert all(b["type"] == "text" for b in blocks)

    def test_table_pdf_returns_table_block(self, tmp_path):
        data = _pdf_with_table(
            headers=["Name", "Amount"],
            rows=[["Alice", "$100"], ["Bob", "$200"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        blocks = PlumberPDFParser().parse_blocks(path)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        assert len(table_blocks) >= 1

    def test_table_block_has_headers_and_rows(self, tmp_path):
        data = _pdf_with_table(
            headers=["Item", "Qty", "Price"],
            rows=[["Widget", "5", "$10"], ["Gadget", "2", "$25"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        blocks = PlumberPDFParser().parse_blocks(path)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        assert table_blocks
        t = table_blocks[0]["table"]
        assert "headers" in t
        assert "rows" in t
        assert isinstance(t["headers"], list)
        assert isinstance(t["rows"], list)

    def test_table_block_text_is_pipe_delimited(self, tmp_path):
        data = _pdf_with_table(
            headers=["Col A", "Col B"],
            rows=[["1", "2"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        blocks = PlumberPDFParser().parse_blocks(path)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        assert table_blocks
        text = table_blocks[0]["text"]
        assert "|" in text
        assert "Col A" in text
        assert "Col B" in text

    def test_blocks_have_page_num(self, tmp_path):
        path = _write_pdf(tmp_path, "a.pdf", _pdf_text_only("Page content."))
        blocks = PlumberPDFParser().parse_blocks(path)
        for b in blocks:
            assert "page_num" in b
            assert b["page_num"] == 1

    def test_file_not_found_raises(self, tmp_path):
        parser = PlumberPDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_blocks(str(tmp_path / "missing.pdf"))

    def test_wrong_extension_raises(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("hello")
        parser = PlumberPDFParser()
        with pytest.raises(ValueError, match=".pdf"):
            parser.parse_blocks(str(p))


# ---------------------------------------------------------------------------
# parse — backward compat (returns page dicts, not blocks)
# ---------------------------------------------------------------------------

class TestParseLegacy:
    def test_parse_returns_list_of_dicts(self, tmp_path):
        path = _write_pdf(tmp_path, "a.pdf", _pdf_text_only("Some content here."))
        pages = PlumberPDFParser().parse(path)
        assert isinstance(pages, list)
        assert all("text" in p for p in pages)

    def test_parse_table_pdf_includes_table_text(self, tmp_path):
        data = _pdf_with_table(
            headers=["Name", "Value"],
            rows=[["Alpha", "100"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pages = PlumberPDFParser().parse(path)
        combined = " ".join(p["text"] for p in pages)
        assert "Name" in combined or "Alpha" in combined


# ---------------------------------------------------------------------------
# Parser ABC — parse_blocks default
# ---------------------------------------------------------------------------

class TestParseBlocksDefault:
    def test_default_parse_blocks_wraps_parse(self, tmp_path):
        """The default parse_blocks() on Parser ABC wraps parse() as type=text."""
        from blue_data_doc_source.source.document.parsers.base import Parser

        class FakeParser(Parser):
            def parse(self, path):
                return [{"page_num": 1, "text": "hello"}, {"page_num": 2, "text": "world"}]

        blocks = FakeParser().parse_blocks("dummy.pdf")
        assert len(blocks) == 2
        assert all(b["type"] == "text" for b in blocks)
        assert blocks[0]["text"] == "hello"
        assert blocks[1]["page_num"] == 2


# ---------------------------------------------------------------------------
# Pipeline integration — content_type on chunks
# ---------------------------------------------------------------------------

class TestPipelineWithPlumber:
    def _make_pipeline(self):
        vs = MemoryVectorSource()
        return DocumentPipeline(
            parser=PlumberPDFParser(),
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
        )

    def test_table_chunks_have_content_type_table(self, tmp_path):
        data = _pdf_with_table(
            headers=["Item", "Price"],
            rows=[["Widget", "$10"], ["Gadget", "$25"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        table_chunks = [r for r in records if r.get("content_type") == "table"]
        assert len(table_chunks) >= 1

    def test_text_chunks_have_content_type_text(self, tmp_path):
        data = _pdf_with_table(
            headers=["A", "B"],
            rows=[["1", "2"]],
            prose_before="This is introductory prose text.",
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        text_chunks = [r for r in records if r.get("content_type") == "text"]
        assert len(text_chunks) >= 1

    def test_table_chunk_has_structured_table_field(self, tmp_path):
        data = _pdf_with_table(
            headers=["Description", "Qty", "Total"],
            rows=[["Widget", "5", "$50"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        table_chunks = [r for r in records if r.get("content_type") == "table"]
        assert table_chunks
        t = table_chunks[0]["table"]
        assert "headers" in t
        assert "rows" in t

    def test_table_chunk_text_is_pipe_delimited(self, tmp_path):
        data = _pdf_with_table(
            headers=["Name", "Score"],
            rows=[["Alice", "95"]],
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        table_chunks = [r for r in records if r.get("content_type") == "table"]
        assert table_chunks
        assert "|" in table_chunks[0]["text"]

    def test_content_type_in_metadata(self, tmp_path):
        data = _pdf_with_table(headers=["A"], rows=[["1"]])
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        for r in records:
            if r.get("content_type"):
                assert r["metadata"].get("content_type") == r["content_type"]

    def test_filter_by_content_type_table(self, tmp_path):
        data = _pdf_with_table(
            headers=["Item", "Price"],
            rows=[["A", "$1"], ["B", "$2"]],
            prose_before="Summary of items sold.",
        )
        path = _write_pdf(tmp_path, "t.pdf", data)
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        pipeline.ingest(path)
        results = pipeline.search("items", k=10, filters={"content_type": "table"})
        assert all(r.get("content_type") == "table" for r in results)

    def test_text_only_pdf_has_no_table_chunks(self, tmp_path):
        path = _write_pdf(tmp_path, "a.pdf", _pdf_text_only("Plain prose. No tables here."))
        pipeline = self._make_pipeline()
        pipeline.ensure_index()
        records = pipeline.ingest(path)
        table_chunks = [r for r in records if r.get("content_type") == "table"]
        assert len(table_chunks) == 0
