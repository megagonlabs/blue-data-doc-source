"""Tests for platform/doc-mcp/src/blue_data_doc_mcp/server.py.

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
_SERVER_PATH = os.path.join(_REPO_ROOT, "platform/doc-mcp/src/blue_data_doc_mcp/server.py")


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
    # The server's configured collection (from BLUE_DOC_COLLECTION); tools resolve
    # an omitted collection against this rather than a hard-coded "documents" (#151).
    m.collection = "documents"
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
    # With no collection argument the tool forwards None, so the pipeline falls back
    # to its configured collection (BLUE_DOC_COLLECTION) — not a hard-coded
    # "documents" — and the response reports the resolved name (#151).
    pipeline = _mock_pipeline(ensure_index=MagicMock())
    pipeline.collection = "my_docs"
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ensure_index())
    pipeline.ensure_index.assert_called_once_with(collection=None)
    assert result["status"] == "ok"
    assert result["collection"] == "my_docs"


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
        collection=None,  # #78: no hard-coded "documents"; a path delete uses the ingested collection
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


def test_build_pipeline_with_redis_constructs_vector_source(monkeypatch):
    # Regression (#91): when BLUE_DOC_REDIS_HOST is set, the builder must construct
    # a RedisVectorSource WITHOUT raising TypeError — connection details go in
    # `properties={"connection": {...}}`, not as host/port kwargs. Construction is
    # lazy (no socket opened), so this stays offline.
    monkeypatch.setenv("BLUE_DOC_REDIS_HOST", "localhost")
    monkeypatch.setenv("BLUE_DOC_REDIS_PORT", "6379")
    pipeline = _server._build_pipeline()
    assert type(pipeline.vector_source).__name__ == "RedisVectorSource"


def _redis_available() -> bool:
    import socket
    s = socket.socket()
    s.settimeout(0.2)
    try:
        return s.connect_ex(("localhost", 6379)) == 0
    finally:
        s.close()


@pytest.mark.skipif(not _redis_available(), reason="Redis not available on localhost:6379")
def test_build_pipeline_with_redis_can_ensure_index(monkeypatch):
    # Regression (#70): the real Redis-backed pipeline must not only build but also
    # create an index — blue-data's create_index requires scan_prefix in the
    # connection properties. Earlier tests patch or skip this path, so the missing
    # scan_prefix went unnoticed (ensure_index raised "requires ... scan_prefix").
    import redis

    monkeypatch.setenv("BLUE_DOC_REDIS_HOST", "localhost")
    monkeypatch.setenv("BLUE_DOC_REDIS_PORT", "6379")
    coll = "test_platform_smoke_70"
    try:
        redis.Redis(host="localhost", port=6379).execute_command("FT.DROPINDEX", coll, "DD")
    except Exception:
        pass
    pipeline = _server._build_pipeline()
    assert type(pipeline.vector_source).__name__ == "RedisVectorSource"
    pipeline.ensure_index(collection=coll)  # must not raise — scan_prefix now present


# ── real-SDK import smoke test (regression for #76) ───────────────────────────

def _load_server_real():
    """Load server.py against the *installed* MCP SDK (no stub)."""
    spec = importlib.util.spec_from_file_location("_blue_doc_mcp_server_real", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_server_imports_against_installed_mcp_sdk(monkeypatch):
    # Regression (#76): a fresh install must import the server against the real MCP
    # SDK. mcp 2.x removed mcp.server.fastmcp, so an unpinned mcp>=1.0.0 resolves to
    # 2.x and `from mcp.server.fastmcp import FastMCP` fails at import. The other
    # tests stub the SDK and never catch this. Skip only when no MCP SDK is present.
    import asyncio

    pytest.importorskip("mcp")
    monkeypatch.delenv("BLUE_DOC_REDIS_HOST", raising=False)
    mod = _load_server_real()  # raises on mcp 2.x (fastmcp removed) → surfaces #76
    tools = {t.name for t in asyncio.run(mod.mcp.list_tools())}
    # the core tools must be registered against the real SDK (more may exist)
    assert {"ingest_document", "search_documents", "ensure_index", "delete_document"} <= tools
    # a tool is callable against the real SDK (no vector store → structured error)
    assert "error" in json.loads(mod.search_documents("q"))


# ── structured + graph tools (regression for #66) ─────────────────────────────

def test_find_documents_returns_records():
    recs = [{"document_id": "doc_a", "document_type": "contract", "fields": {}}]
    pipeline = _mock_pipeline(find_records=MagicMock(return_value=recs))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.find_documents({"document_type": "contract"}))
    assert result["count"] == 1
    assert result["records"][0]["document_type"] == "contract"
    pipeline.find_records.assert_called_once_with({"document_type": "contract"})


def test_find_documents_respects_limit():
    recs = [{"document_id": f"doc_{i}"} for i in range(10)]
    pipeline = _mock_pipeline(find_records=MagicMock(return_value=recs))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.find_documents(limit=3))
    assert result["count"] == 3
    assert len(result["records"]) == 3


def test_find_then_search_delegates_to_pipeline_merge():
    # The tool must call the shared DocumentPipeline.find_then_search merge, not
    # re-implement it (#147).
    chunk = _make_chunk(doc_id="doc_a", idx=0)
    chunk["score"] = 0.9
    pipeline = _mock_pipeline(find_then_search=MagicMock(return_value=[chunk]))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.find_then_search("termination", {"document_type": "contract"}, k=3))
    pipeline.find_then_search.assert_called_once()
    assert result["count"] == 1
    assert result["results"][0]["chunk_id"] == "doc_a_chunk_000000"


def test_find_then_search_no_matches_returns_empty():
    pipeline = _mock_pipeline(find_then_search=MagicMock(return_value=[]))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.find_then_search("x", {"document_type": "nope"}))
    assert result == {"results": [], "count": 0}


def test_find_then_search_ranks_nearest_first_real_backend(tmp_path):
    # #147: real MemoryVectorSource + deterministic embedding — the MCP tool must
    # return the nearest chunk first and match DocumentPipeline.find_then_search.
    from blue_data.source.vector.memory_vector_source import MemoryVectorSource
    from blue_data_doc_source import DocumentPipeline, EmbeddingModel, KeywordClassifier

    class _Toy(EmbeddingModel):
        dim = 2

        def embed_text(self, text):
            return [float(text.lower().count("termination")), 1.0]

    pipeline = DocumentPipeline(
        embedding_model=_Toy(),
        vector_source=MemoryVectorSource(name="m"),
        classifier=KeywordClassifier(),
    )
    pipeline.ensure_index()
    base = "SERVICES AGREEMENT between the parties. Governing law. Indemnification. "
    pipeline.ingest_text(base + "termination " * 5, file_name="far.txt")
    pipeline.ingest_text(base + "termination once", file_name="nearest.txt")

    find_filter = {"document_type": "contract"}
    with patch.object(_server, "_pipeline", pipeline):
        mcp = json.loads(_server.find_then_search("termination", find_filter, k=1))["results"]
    lib = pipeline.find_then_search("termination", find_filter=find_filter, k=1)

    assert mcp[0]["file_name"] == "nearest.txt"
    assert [h["file_name"] for h in mcp] == [h["file_name"] for h in lib]


def test_get_document_record_returns_record():
    pipeline = _mock_pipeline(
        get_record=MagicMock(return_value={"document_id": "doc_a", "document_type": "contract"})
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.get_document_record("doc_a"))
    assert result["document_type"] == "contract"


def test_get_document_record_missing_returns_error():
    pipeline = _mock_pipeline(get_record=MagicMock(return_value=None))
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.get_document_record("nope"))
    assert "error" in result


def test_graph_neighbors_returns_neighbors():
    pipeline = _mock_pipeline(
        graph_neighbors=MagicMock(return_value=[{"chunk_id": "c1", "entity": "Acme"}])
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.graph_neighbors("Acme"))
    assert result["count"] == 1


def test_graph_neighbors_without_graph_returns_error():
    pipeline = _mock_pipeline(
        graph_neighbors=MagicMock(side_effect=ValueError("graph_neighbors requires a graph_source"))
    )
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.graph_neighbors("Acme"))
    assert "error" in result


# ── #151: tools honour BLUE_DOC_COLLECTION (no hard-coded "documents") ──────────

def test_collection_defaults_are_none_in_schema():
    # The four tools must declare collection=None so an omitted argument falls back
    # to the server's configured collection — not a hard-coded "documents" (#151).
    import inspect
    for tool in (_server.ingest_document, _server.search_documents,
                 _server.ensure_index, _server.find_then_search):
        assert inspect.signature(tool).parameters["collection"].default is None


def test_ingest_omitted_collection_uses_configured():
    pipeline = _mock_pipeline(ingest=MagicMock(return_value=[_make_chunk()]))
    pipeline.collection = "my_docs"
    with patch.object(_server, "_pipeline", pipeline):
        result = json.loads(_server.ingest_document("report.pdf"))
    pipeline.ingest.assert_called_once_with("report.pdf", collection=None)
    assert result["collection"] == "my_docs"


def test_search_omitted_collection_forwards_none():
    pipeline = _mock_pipeline(search=MagicMock(return_value=[]))
    pipeline.collection = "my_docs"
    with patch.object(_server, "_pipeline", pipeline):
        _server.search_documents("query")
    pipeline.search.assert_called_once_with("query", k=5, collection=None)


def test_find_then_search_omitted_collection_forwards_none():
    pipeline = _mock_pipeline(find_then_search=MagicMock(return_value=[]))
    pipeline.collection = "my_docs"
    with patch.object(_server, "_pipeline", pipeline):
        _server.find_then_search("query")
    _, kwargs = pipeline.find_then_search.call_args
    assert kwargs["collection"] is None
