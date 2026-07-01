import pytest
from unittest.mock import MagicMock
from blue_doc_source.source.document.pipeline import DocumentPipeline
from blue_doc_source.source.document.embeddings import DummyEmbeddingModel
from blue_doc_source.source.document.enrichment import BasicEnricher


def make_txt(tmp_path, content="The quick brown fox jumps over the lazy dog"):
    f = tmp_path / "test.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


def test_ingest_txt_returns_records(tmp_path):
    path = make_txt(tmp_path)
    pipeline = DocumentPipeline(chunk_size=5, overlap=1)
    records = pipeline.ingest(path)
    assert len(records) > 0


def test_ingest_txt_record_fields(tmp_path):
    path = make_txt(tmp_path)
    pipeline = DocumentPipeline(chunk_size=5, overlap=1)
    record = pipeline.ingest(path)[0]
    assert record["file_name"] == "test.txt"
    assert record["source_type"] == "document"
    assert record["mime_type"] == "text/plain"
    assert "text" in record
    assert "document_id" in record
    assert "chunk_id" in record


def test_ingest_txt_has_ingested_at(tmp_path):
    path = make_txt(tmp_path)
    records = DocumentPipeline(chunk_size=5, overlap=1).ingest(path)
    assert "ingested_at" in records[0]["metadata"]


def test_ingest_txt_with_embedding(tmp_path):
    path = make_txt(tmp_path)
    pipeline = DocumentPipeline(chunk_size=5, overlap=1, embedding_model=DummyEmbeddingModel(dim=8))
    records = pipeline.ingest(path)
    assert "embedding" in records[0]
    assert len(records[0]["embedding"]) == 8


def test_ingest_txt_upserts_to_vector_source(tmp_path):
    path = make_txt(tmp_path)
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    records = pipeline.ingest(path)
    assert vector_source.upsert.call_count == len(records)


def test_ingest_no_upsert_without_embedding(tmp_path):
    path = make_txt(tmp_path)
    vector_source = MagicMock()
    pipeline = DocumentPipeline(chunk_size=5, overlap=1, vector_source=vector_source)
    pipeline.ingest(path)
    vector_source.upsert.assert_not_called()


def test_ingest_unsupported_extension_raises(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentPipeline().ingest(str(f))


def test_search_requires_embedding_model(tmp_path):
    vector_source = MagicMock()
    pipeline = DocumentPipeline(vector_source=vector_source)
    with pytest.raises(ValueError, match="embedding_model"):
        pipeline.search("query")


def test_search_requires_vector_source():
    pipeline = DocumentPipeline(embedding_model=DummyEmbeddingModel())
    with pytest.raises(ValueError, match="vector_source"):
        pipeline.search("query")


def test_search_calls_vector_source():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    pipeline.search("what is eligibility?", k=3)
    vector_source.knn.assert_called_once()
    call_kwargs = vector_source.knn.call_args
    assert call_kwargs.kwargs["k"] == 3


def test_enrichment_populates_entities_and_hints(tmp_path):
    path = make_txt(tmp_path, "Contact John Smith at john@example.com by January 2024.")
    pipeline = DocumentPipeline(chunk_size=50, overlap=5, enricher=BasicEnricher())
    record = pipeline.ingest(path)[0]
    assert isinstance(record["entities"], list)
    assert isinstance(record["semantic_hints"], dict)
    assert "keywords" in record["semantic_hints"]


def test_no_enricher_leaves_entities_empty(tmp_path):
    path = make_txt(tmp_path)
    record = DocumentPipeline(chunk_size=50, overlap=5).ingest(path)[0]
    assert record["entities"] == []
    assert record["semantic_hints"] == {}


def test_all_chunks_share_document_id(tmp_path):
    content = " ".join(["word"] * 50)
    path = make_txt(tmp_path, content)
    records = DocumentPipeline(chunk_size=5, overlap=1).ingest(path)
    doc_ids = {r["document_id"] for r in records}
    assert len(doc_ids) == 1


def test_same_file_same_document_id(tmp_path):
    path = make_txt(tmp_path, " ".join(["word"] * 50))
    pipeline = DocumentPipeline(chunk_size=5, overlap=1)
    first = pipeline.ingest(path)
    second = pipeline.ingest(path)
    assert first[0]["document_id"] == second[0]["document_id"]


def test_different_content_different_document_id(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("content alpha", encoding="utf-8")
    p2.write_text("content beta", encoding="utf-8")
    pipeline = DocumentPipeline(chunk_size=50, overlap=0)
    id1 = pipeline.ingest(str(p1))[0]["document_id"]
    id2 = pipeline.ingest(str(p2))[0]["document_id"]
    assert id1 != id2


def test_ingest_collection_override(tmp_path):
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        chunk_size=5,
        overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        collection="default",
    )
    pipeline.ingest(make_txt(tmp_path), collection="override")
    assert vector_source.upsert.call_args_list[0].kwargs["collection"] == "override"


def test_search_collection_override():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        collection="default",
    )
    pipeline.search("query", collection="override")
    assert vector_source.knn.call_args.kwargs["index_name"] == "override"
