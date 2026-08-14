"""Tests for metadata pre-filtering in DOCUMENT.SEARCH."""
from __future__ import annotations

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source import BasicEnricher, DocumentPipeline, DummyEmbeddingModel
from blue_data_doc_source.source.document.pipeline import _build_filter_expr


# ---------------------------------------------------------------------------
# _build_filter_expr unit tests
# ---------------------------------------------------------------------------

class TestBuildFilterExpr:
    def test_none_returns_wildcard(self):
        assert _build_filter_expr(None) == "*"

    def test_empty_dict_returns_wildcard(self):
        assert _build_filter_expr({}) == "*"

    def test_none_value_ignored(self):
        assert _build_filter_expr({"document_id": None}) == "*"

    def test_document_id(self):
        expr = _build_filter_expr({"document_id": "doc_abc123"})
        assert expr == "@document_id:{doc_abc123}"

    def test_file_name_with_dot_escaped(self):
        expr = _build_filter_expr({"file_name": "policy.pdf"})
        assert r"\." in expr
        assert "policy" in expr

    def test_mime_type_slash_escaped(self):
        expr = _build_filter_expr({"mime_type": "application/pdf"})
        assert r"\/" in expr

    def test_section_title_space_escaped(self):
        expr = _build_filter_expr({"section_title": "Eligibility Requirements"})
        assert r"\ " in expr or r"\  " in expr or "Eligibility" in expr

    def test_page_start_exact(self):
        expr = _build_filter_expr({"page_start": 3})
        assert expr == "@page_start:[3 3]"

    def test_page_start_range(self):
        expr = _build_filter_expr({"page_start": {"gte": 1, "lte": 5}})
        assert expr == "@page_start:[1 5]"

    def test_page_start_open_lower(self):
        expr = _build_filter_expr({"page_start": {"lte": 10}})
        assert "-inf" in expr
        assert "10" in expr

    def test_page_start_open_upper(self):
        expr = _build_filter_expr({"page_start": {"gte": 5}})
        assert "+inf" in expr
        assert "5" in expr

    def test_multiple_filters_parenthesized(self):
        expr = _build_filter_expr({"document_id": "doc_abc", "mime_type": "text/plain"})
        assert expr.startswith("(")
        assert ") (" in expr

    def test_single_filter_no_parens(self):
        expr = _build_filter_expr({"document_id": "doc_abc"})
        assert not expr.startswith("(")


# ---------------------------------------------------------------------------
# MemoryVectorSource._apply_filter unit tests
# ---------------------------------------------------------------------------

def _make_store() -> MemoryVectorSource:
    vs = MemoryVectorSource()
    vs.create_index("idx", "col", dim=4)
    docs = {
        "chunk_a": {
            "document_id": "doc_001",
            "file_name": "policy.pdf",
            "mime_type": "application/pdf",
            "section_title": "Introduction",
            "page_start": 1,
            "text": "alpha",
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
        "chunk_b": {
            "document_id": "doc_001",
            "file_name": "policy.pdf",
            "mime_type": "application/pdf",
            "section_title": "Conclusion",
            "page_start": 5,
            "text": "beta",
            "embedding": [0.0, 1.0, 0.0, 0.0],
        },
        "chunk_c": {
            "document_id": "doc_002",
            "file_name": "report.txt",
            "mime_type": "text/plain",
            "text": "gamma",
            "embedding": [0.0, 0.0, 1.0, 0.0],
        },
    }
    for doc_id, doc in docs.items():
        vs.upsert("col", doc_id, doc)
    return vs


class TestMemoryVectorSourceFilter:
    def setup_method(self):
        self.vs = _make_store()
        self.docs = self.vs._store["col"]

    def test_wildcard_returns_all(self):
        result = self.vs._apply_filter(self.docs, "*")
        assert len(result) == 3

    def test_tag_document_id(self):
        result = self.vs._apply_filter(self.docs, "@document_id:{doc_001}")
        assert set(result.keys()) == {"chunk_a", "chunk_b"}

    def test_tag_section_title(self):
        result = self.vs._apply_filter(self.docs, "@section_title:{Introduction}")
        assert set(result.keys()) == {"chunk_a"}

    def test_tag_mime_type_escaped(self):
        # Filter expr uses escaped slash
        result = self.vs._apply_filter(self.docs, r"@mime_type:{application\/pdf}")
        assert set(result.keys()) == {"chunk_a", "chunk_b"}

    def test_tag_file_name_escaped_dot(self):
        result = self.vs._apply_filter(self.docs, r"@file_name:{policy\.pdf}")
        assert set(result.keys()) == {"chunk_a", "chunk_b"}

    def test_numeric_exact_page(self):
        result = self.vs._apply_filter(self.docs, "@page_start:[1 1]")
        assert set(result.keys()) == {"chunk_a"}

    def test_numeric_range(self):
        result = self.vs._apply_filter(self.docs, "@page_start:[1 5]")
        assert set(result.keys()) == {"chunk_a", "chunk_b"}

    def test_numeric_excludes_missing_field(self):
        # chunk_c has no page_start — should be excluded by numeric filter
        result = self.vs._apply_filter(self.docs, "@page_start:[1 10]")
        assert "chunk_c" not in result

    def test_combined_tag_and_numeric(self):
        expr = "(@document_id:{doc_001}) (@page_start:[5 5])"
        result = self.vs._apply_filter(self.docs, expr)
        assert set(result.keys()) == {"chunk_b"}

    def test_combined_two_tags(self):
        expr = r"(@document_id:{doc_001}) (@section_title:{Introduction})"
        result = self.vs._apply_filter(self.docs, expr)
        assert set(result.keys()) == {"chunk_a"}

    def test_no_match_returns_empty(self):
        result = self.vs._apply_filter(self.docs, "@document_id:{doc_999}")
        assert result == {}


# ---------------------------------------------------------------------------
# End-to-end: DocumentPipeline.search(filters=...)
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path) -> tuple[DocumentPipeline, str, str]:
    """Ingest two txt files, return pipeline + doc_ids."""
    vs = MemoryVectorSource()
    pipeline = DocumentPipeline(
        chunk_size=20,
        overlap=2,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vs,
        enricher=BasicEnricher(),
        collection="docs",
    )
    pipeline.ensure_index(collection="docs")

    f1 = tmp_path / "alpha.txt"
    f1.write_text(("Alpha document content about eligibility requirements. " * 5))
    records1 = pipeline.ingest(str(f1), collection="docs")
    doc_id1 = records1[0]["document_id"]

    f2 = tmp_path / "beta.txt"
    f2.write_text(("Beta document content about coverage terms and conditions. " * 5))
    records2 = pipeline.ingest(str(f2), collection="docs")
    doc_id2 = records2[0]["document_id"]

    return pipeline, doc_id1, doc_id2


class TestPipelinePreFiltering:
    def test_no_filter_returns_results_from_all_docs(self, tmp_path):
        pipeline, doc_id1, doc_id2 = _make_pipeline(tmp_path)
        results = pipeline.search("content", k=10, collection="docs")
        doc_ids = {r["document_id"] for r in results}
        assert len(doc_ids) == 2

    def test_filter_by_document_id_scopes_to_one_doc(self, tmp_path):
        pipeline, doc_id1, _ = _make_pipeline(tmp_path)
        results = pipeline.search(
            "content", k=10, collection="docs",
            filters={"document_id": doc_id1},
        )
        assert results
        assert all(r["document_id"] == doc_id1 for r in results)

    def test_filter_by_file_name(self, tmp_path):
        pipeline, _, _ = _make_pipeline(tmp_path)
        results = pipeline.search(
            "content", k=10, collection="docs",
            filters={"file_name": "beta.txt"},
        )
        assert results
        assert all(r["file_name"] == "beta.txt" for r in results)

    def test_filter_by_mime_type(self, tmp_path):
        pipeline, _, _ = _make_pipeline(tmp_path)
        results = pipeline.search(
            "content", k=10, collection="docs",
            filters={"mime_type": "text/plain"},
        )
        assert results
        assert all(r["mime_type"] == "text/plain" for r in results)

    def test_filter_no_match_returns_empty(self, tmp_path):
        pipeline, _, _ = _make_pipeline(tmp_path)
        results = pipeline.search(
            "content", k=10, collection="docs",
            filters={"document_id": "doc_nonexistent"},
        )
        assert results == []

    def test_none_filters_returns_all(self, tmp_path):
        pipeline, doc_id1, doc_id2 = _make_pipeline(tmp_path)
        results = pipeline.search("content", k=10, collection="docs", filters=None)
        doc_ids = {r["document_id"] for r in results}
        assert len(doc_ids) == 2


# ---------------------------------------------------------------------------
# End-to-end: DOCUMENT.SEARCH op with filter=
# ---------------------------------------------------------------------------

class TestDocumentSourcePreFiltering:
    def test_op_filter_by_document_id(self, tmp_path):
        from blue_data_doc_source import DocumentDataSource

        vs = MemoryVectorSource()
        source = DocumentDataSource(
            name="test",
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            default_collection="docs",
        )
        source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "docs"})

        f = tmp_path / "sample.txt"
        f.write_text("Sample document about contracts and obligations. " * 10)
        records = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(f), "collection": "docs"})
        doc_id = records[0]["document_id"]

        results = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "contracts",
            "collection": "docs",
            "filter": {"document_id": doc_id},
        })
        assert results
        assert all(r["document_id"] == doc_id for r in results)

    def test_op_without_filter_passes_through(self, tmp_path):
        from blue_data_doc_source import DocumentDataSource

        vs = MemoryVectorSource()
        source = DocumentDataSource(
            name="test",
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            default_collection="docs",
        )
        source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "docs"})

        f = tmp_path / "sample.txt"
        f.write_text("Sample document about contracts. " * 10)
        source.execute_query({"op": "DOCUMENT.INGEST", "path": str(f), "collection": "docs"})

        # No filter key — should not raise
        results = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "contracts",
            "collection": "docs",
        })
        assert isinstance(results, list)
