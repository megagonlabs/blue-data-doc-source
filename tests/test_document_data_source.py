import pytest
from unittest.mock import MagicMock

from blue_doc_source.source import DocumentDataSource
from blue_doc_source.embeddings import DummyEmbeddingModel
from blue_doc_source.enrichment import BasicEnricher


def make_txt(tmp_path, content="The quick brown fox jumps over the lazy dog"):
    f = tmp_path / "test.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


def test_ingest_op_returns_records(tmp_path):
    source = DocumentDataSource(chunk_size=5, overlap=1)
    records = source.execute_query({"op": "DOCUMENT.INGEST", "path": make_txt(tmp_path)})
    assert isinstance(records, list)
    assert len(records) > 0


def test_ingest_op_missing_path_raises():
    source = DocumentDataSource()
    with pytest.raises(ValueError, match="path"):
        source.execute_query({"op": "DOCUMENT.INGEST"})


def test_ingest_op_uses_collection(tmp_path):
    vector_store = MagicMock()
    source = DocumentDataSource(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_store=vector_store,
    )
    source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": make_txt(tmp_path),
        "collection": "my_collection",
    })
    call_kwargs = vector_store.upsert.call_args_list[0]
    assert call_kwargs.kwargs["collection"] == "my_collection"


def test_search_op_calls_vector_store():
    vector_store = MagicMock()
    vector_store.search.return_value = []
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_store=vector_store,
    )
    results = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "eligibility requirements",
        "k": 3,
    })
    vector_store.search.assert_called_once()
    assert results == []


def test_search_op_missing_query_raises():
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(),
        vector_store=MagicMock(),
    )
    with pytest.raises(ValueError, match="query"):
        source.execute_query({"op": "DOCUMENT.SEARCH"})


def test_search_op_passes_k():
    vector_store = MagicMock()
    vector_store.search.return_value = []
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_store=vector_store,
    )
    source.execute_query({"op": "DOCUMENT.SEARCH", "query": "test", "k": 7})
    assert vector_store.search.call_args.kwargs["k"] == 7


def test_unsupported_op_raises():
    source = DocumentDataSource()
    with pytest.raises(ValueError, match="Unsupported op"):
        source.execute_query({"op": "DOCUMENT.DELETE"})


def test_ingest_with_enricher(tmp_path):
    source = DocumentDataSource(
        chunk_size=50,
        overlap=5,
        enricher=BasicEnricher(),
    )
    records = source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": make_txt(tmp_path, "Contact John Smith at john@example.com"),
    })
    assert "entities" in records[0]
    assert "semantic_hints" in records[0]
