"""Tests for platform/doc-mcp/src/server.py.

The mcp package is stubbed out so these tests run without the MCP SDK on PATH
and without any network or Redis connection. The pipeline singleton is patched
via unittest.mock so each test runs against a fresh mock.
"""
import importlib.util
import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ── Load server module with mcp stubbed out ───────────────────────────────────

def _stub_mcp():
    """Return a sys.modules patch dict that replaces the mcp package with stubs."""
    stub_mcp = types.ModuleType("mcp")
    stub_server = types.ModuleType("mcp.server")
    stub_fastmcp_mod = types.ModuleType("mcp.server.fastmcp")

    class _FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self, **kwargs):
            return lambda fn: fn

        def run(self, transport="stdio"):
            pass

    stub_fastmcp_mod.FastMCP = _FakeFastMCP
    stub_mcp.server = stub_server
    stub_server.fastmcp = stub_fastmcp_mod
    return {"mcp": stub_mcp, "mcp.server": stub_server, "mcp.server.fastmcp": stub_fastmcp_mod}


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_SERVER_PATH = os.path.join(_REPO_ROOT, "platform/doc-mcp/src/server.py")


def _load_server():
    with patch.dict(sys.modules, _stub_mcp()):
        spec = importlib.util.spec_from_file_location("_blue_doc_mcp_server", _SERVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


# Load once for the module; each test patches _pipeline independently.
_server = _load_server()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(doc_id="doc_abc123", idx=0, section=None, page=None):
    c = {
        "document_id": doc_id,
        "chunk_id": f"{doc_id}_chunk_{idx:06d}",
        "file_name": "test.pdf",
        "text": f"Chunk {idx} text content.",
    }
    if section:
        c["section_title"] = section
    if page is not None:
        c["page_start"] = page
    return c


def _mock_pipeline(**kwargs):
    m = MagicMock()
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


# ── ingest_document ───────────────────────────────────────────────────────────

def test_ingest_returns_document_id_and_chunk_count():
    chunks = [_make_chunk(idx=i) for i in range(3)]
    pipeline = _mock_pipeline(ingest=MagicMock(return_value=chunks))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ingest_document("report.pdf"))
    assert result["document_id"] == "doc_abc123"
    assert result["chunk_count"] == 3
    assert result["file_name"] == "test.pdf"
    assert result["collection"] == "documents"


def test_ingest_passes_collection_to_pipeline():
    chunks = [_make_chunk()]
    pipeline = _mock_pipeline(ingest=MagicMock(return_value=chunks))
    with patch.object(_server, "_pipeline", pipeline):
        _server.ingest_document("report.pdf", collection="contracts")
    pipeline.ingest.assert_called_once_with("report.pdf", collection="contracts")


def test_ingest_empty_file_returns_error():
    pipeline = _mock_pipeline(ingest=MagicMock(return_value=[]))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ingest_document("empty.txt"))
    assert "error" in result


def test_ingest_file_not_found_returns_error():
    pipeline = _mock_pipeline(ingest=MagicMock(side_effect=FileNotFoundError("not found")))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ingest_document("/no/such/file.pdf"))
    assert "error" in result
    assert "not found" in result["error"].lower() or "/no/such/file.pdf" in result["error"]


def test_ingest_unsupported_extension_returns_error():
    pipeline = _mock_pipeline(ingest=MagicMock(side_effect=ValueError("Unsupported file type: .pptx")))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ingest_document("deck.pptx"))
    assert "error" in result
    assert "Unsupported" in result["error"]


# ── search_documents ──────────────────────────────────────────────────────────

def test_search_returns_hits():
    chunks = [
        _make_chunk(idx=0, section="Introduction", page=1),
        _make_chunk(idx=1),
    ]
    pipeline = _mock_pipeline(search=MagicMock(return_value=chunks))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.search_documents("eligibility requirements"))
    assert result["count"] == 2
    assert result["results"][0]["section_title"] == "Introduction"
    assert result["results"][0]["page"] == 1
    assert "section_title" not in result["results"][1]
    assert "page" not in result["results"][1]


def test_search_passes_k_and_collection():
    pipeline = _mock_pipeline(search=MagicMock(return_value=[]))
    with patch.object(_server, "_pipeline", pipeline):
        _server.search_documents("query", collection="contracts", k=10)
    pipeline.search.assert_called_once_with("query", k=10, collection="contracts")


def test_search_no_embedding_model_returns_error():
    pipeline = _mock_pipeline(
        search=MagicMock(side_effect=ValueError("An embedding_model is required for search"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.search_documents("query"))
    assert "error" in result
    assert "embedding_model" in result["error"]


def test_search_no_vector_source_returns_error():
    pipeline = _mock_pipeline(
        search=MagicMock(side_effect=ValueError("A vector_source is required for search"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.search_documents("query"))
    assert "error" in result


# ── ensure_index ──────────────────────────────────────────────────────────────

def test_ensure_index_calls_pipeline_and_returns_ok():
    pipeline = _mock_pipeline(ensure_index=MagicMock())
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ensure_index("contracts"))
    pipeline.ensure_index.assert_called_once_with(collection="contracts")
    assert result["status"] == "ok"
    assert result["collection"] == "contracts"


def test_ensure_index_default_collection():
    pipeline = _mock_pipeline(ensure_index=MagicMock())
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ensure_index())
    pipeline.ensure_index.assert_called_once_with(collection="documents")
    assert result["status"] == "ok"


def test_ensure_index_error_returns_error():
    pipeline = _mock_pipeline(
        ensure_index=MagicMock(side_effect=Exception("Redis unavailable"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ensure_index())
    assert "error" in result
    assert "Redis" in result["error"]


# ── delete_document ───────────────────────────────────────────────────────────

def test_delete_by_path_returns_deleted_count():
    pipeline = _mock_pipeline(delete=MagicMock(return_value=5))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.delete_document(path="/docs/report.pdf"))
    assert result["deleted"] == 5
    pipeline.delete.assert_called_once_with(
        path="/docs/report.pdf",
        document_id=None,
        chunk_count=None,
        collection="documents",
    )


def test_delete_by_document_id_and_chunk_count():
    pipeline = _mock_pipeline(delete=MagicMock(return_value=3))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(
            _server.delete_document(document_id="doc_abc123", chunk_count=3, collection="contracts")
        )
    assert result["deleted"] == 3
    pipeline.delete.assert_called_once_with(
        path=None,
        document_id="doc_abc123",
        chunk_count=3,
        collection="contracts",
    )


def test_delete_unknown_path_returns_error():
    pipeline = _mock_pipeline(
        delete=MagicMock(side_effect=ValueError("No registry entry for '/unknown.pdf'"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.delete_document(path="/unknown.pdf"))
    assert "error" in result
    assert "registry" in result["error"]


def test_delete_missing_args_returns_error():
    pipeline = _mock_pipeline(
        delete=MagicMock(side_effect=ValueError("Provide either 'path' or both 'document_id' and 'chunk_count'"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.delete_document())
    assert "error" in result


# ── _build_pipeline env-var wiring ────────────────────────────────────────────

def test_build_pipeline_uses_dummy_embedding_by_default(monkeypatch):
    monkeypatch.delenv("BLUE_DOC_EMBEDDING", raising=False)
    monkeypatch.delenv("BLUE_DOC_REDIS_HOST", raising=False)
    from blue_data_doc_source import DummyEmbeddingModel
    pipeline = _server._build_pipeline()
    assert isinstance(pipeline.embedding_model, DummyEmbeddingModel)


def test_build_pipeline_ocr_fallback_env(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_OCR_FALLBACK", "true")
    monkeypatch.delenv("BLUE_DOC_REDIS_HOST", raising=False)
    pipeline = _server._build_pipeline()
    assert pipeline.pdf_parser._ocr_fallback is True


def test_build_pipeline_chunk_size_env(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_CHUNK_SIZE", "500")
    monkeypatch.setenv("BLUE_DOC_OVERLAP", "75")
    monkeypatch.delenv("BLUE_DOC_REDIS_HOST", raising=False)
    pipeline = _server._build_pipeline()
    assert pipeline.chunk_size == 500
    assert pipeline.overlap == 75
