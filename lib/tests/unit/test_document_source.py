import pytest
from unittest.mock import MagicMock

from blue_doc_source.source.document.document_source import DocumentDataSource
from blue_doc_source.source.document.embeddings import DummyEmbeddingModel
from blue_doc_source.source.document.enrichment import BasicEnricher


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
    vector_source = MagicMock()
    source = DocumentDataSource(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": make_txt(tmp_path),
        "collection": "my_collection",
    })
    call_kwargs = vector_source.upsert.call_args_list[0]
    assert call_kwargs.kwargs["collection"] == "my_collection"


def test_search_op_calls_vector_source():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    results = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "eligibility requirements",
        "k": 3,
    })
    vector_source.knn.assert_called_once()
    assert results == []


def test_search_op_missing_query_raises():
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(),
        vector_source=MagicMock(),
    )
    with pytest.raises(ValueError, match="query"):
        source.execute_query({"op": "DOCUMENT.SEARCH"})


def test_search_op_passes_k():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    source.execute_query({"op": "DOCUMENT.SEARCH", "query": "test", "k": 7})
    assert vector_source.knn.call_args.kwargs["k"] == 7


def test_unsupported_op_raises():
    source = DocumentDataSource()
    with pytest.raises(ValueError, match="Unsupported op"):
        source.execute_query({"op": "DOCUMENT.UNKNOWN"})


def test_delete_op_removes_chunks(tmp_path):
    vector_source = MagicMock()
    source = DocumentDataSource(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    path = make_txt(tmp_path)
    records = source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    deleted = source.execute_query({"op": "DOCUMENT.DELETE", "path": path})
    assert deleted == len(records)
    assert vector_source.delete.call_count == len(records)


def test_delete_op_cross_process(tmp_path):
    vector_source = MagicMock()
    source = DocumentDataSource(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    path = make_txt(tmp_path)
    records = source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    doc_id = records[0]["document_id"]
    deleted = source.execute_query({
        "op": "DOCUMENT.DELETE",
        "document_id": doc_id,
        "chunk_count": len(records),
    })
    assert deleted == len(records)


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


def test_pipeline_reused_across_calls(tmp_path):
    source = DocumentDataSource(chunk_size=5, overlap=1)
    pipeline_id = id(source._pipeline)
    source.execute_query({"op": "DOCUMENT.INGEST", "path": make_txt(tmp_path)})
    assert id(source._pipeline) == pipeline_id


def test_deduplication_same_file_same_document_id(tmp_path):
    source = DocumentDataSource(chunk_size=5, overlap=1)
    path = make_txt(tmp_path)
    first = source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    second = source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    assert first[0]["document_id"] == second[0]["document_id"]


def test_ensure_index_op_calls_create_index():
    vector_source = MagicMock()
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        default_collection="docs",
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX"})
    vector_source.create_index.assert_called_once()


def test_ensure_index_op_collection_override():
    vector_source = MagicMock()
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})
    assert vector_source.create_index.call_args.kwargs["index_name"] == "contracts"
