import pytest
from unittest.mock import MagicMock, patch
from blue_doc_source.source.document.pipeline import DocumentPipeline
from blue_doc_source.source.document.embeddings import DummyEmbeddingModel, EmbeddingModel
from blue_doc_source.source.document.chunkers import SectionChunker, WindowChunker
from blue_doc_source.source.document.enrichment import BasicEnricher
from typing import List


class _WrongDimModel(EmbeddingModel):
    """Returns vectors shorter than advertised dim — for mismatch validation tests."""
    @property
    def dim(self) -> int:
        return 16

    def embed_text(self, text: str) -> List[float]:
        return [0.0] * 4  # intentionally wrong


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


def test_section_title_in_record(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("# Eligibility\nCustomers must meet the following criteria.", encoding="utf-8")
    pipeline = DocumentPipeline(chunk_size=200, overlap=0)
    records = pipeline.ingest(str(f))
    titled = [r for r in records if r.get("section_title") == "Eligibility"]
    assert len(titled) > 0


def test_plain_text_section_title_absent(tmp_path):
    path = make_txt(tmp_path, "no headers here just plain words")
    records = DocumentPipeline(chunk_size=200, overlap=0).ingest(path)
    assert all("section_title" not in r for r in records)


def test_ensure_index_calls_create_index():
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        collection="docs",
    )
    pipeline.ensure_index()
    vector_source.create_index.assert_called_once_with(
        index_name="docs", collection="docs", dim=8
    )


def test_ensure_index_collection_override():
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        collection="default",
    )
    pipeline.ensure_index(collection="override")
    assert vector_source.create_index.call_args.kwargs["index_name"] == "override"


def test_ensure_index_no_op_without_vector_source():
    pipeline = DocumentPipeline(embedding_model=DummyEmbeddingModel(dim=8))
    pipeline.ensure_index()  # should not raise


def test_ensure_index_no_op_without_embedding_model():
    pipeline = DocumentPipeline(vector_source=MagicMock())
    pipeline.ensure_index()  # should not raise


def test_embedding_dim_property():
    assert DummyEmbeddingModel(dim=16).dim == 16


def test_ensure_index_idempotent_no_double_create():
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        collection="docs",
    )
    pipeline.ensure_index()
    pipeline.ensure_index()
    assert vector_source.create_index.call_count == 1


def test_ensure_index_suppresses_already_exists_error():
    vector_source = MagicMock()
    vector_source.create_index.side_effect = Exception("Index already exists")
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    pipeline.ensure_index()  # should not raise


def test_ensure_index_reraises_other_errors():
    vector_source = MagicMock()
    vector_source.create_index.side_effect = Exception("Connection refused")
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    with pytest.raises(Exception, match="Connection refused"):
        pipeline.ensure_index()


def test_stale_chunks_deleted_on_content_change(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("original content", encoding="utf-8")
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        chunk_size=50,
        overlap=0,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    pipeline.ingest(str(path))
    first_call_count = vector_source.upsert.call_count

    path.write_text("completely different content that changes the hash", encoding="utf-8")
    pipeline.ingest(str(path))

    vector_source.delete.assert_called()
    assert vector_source.upsert.call_count > first_call_count


def test_same_content_no_delete(tmp_path):
    path = make_txt(tmp_path)
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        chunk_size=50,
        overlap=0,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    pipeline.ingest(str(path))
    pipeline.ingest(str(path))
    vector_source.delete.assert_not_called()


def test_embedding_dim_mismatch_raises(tmp_path):
    path = make_txt(tmp_path)
    pipeline = DocumentPipeline(chunk_size=50, overlap=0, embedding_model=_WrongDimModel())
    with pytest.raises(ValueError, match="dimension mismatch"):
        pipeline.ingest(path)


def test_default_chunker_is_section_chunker():
    pipeline = DocumentPipeline()
    assert isinstance(pipeline.chunker, SectionChunker)


@patch("blue_doc_source.source.document.pipeline.PDFParser")
def test_ingest_pdf_returns_records(mock_parser_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_parser_cls.return_value.parse_pages.return_value = [
        {"page_num": 1, "text": "Page one content here."},
        {"page_num": 2, "text": "Page two content here."},
    ]
    pipeline = DocumentPipeline(chunk_size=200, overlap=0)
    records = pipeline.ingest(str(pdf_path))
    assert len(records) >= 2
    assert records[0]["mime_type"] == "application/pdf"


@patch("blue_doc_source.source.document.pipeline.PDFParser")
def test_ingest_pdf_page_provenance(mock_parser_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_parser_cls.return_value.parse_pages.return_value = [
        {"page_num": 3, "text": "Content on page three."},
    ]
    pipeline = DocumentPipeline(chunk_size=200, overlap=0)
    records = pipeline.ingest(str(pdf_path))
    assert records[0]["page_start"] == 3
    assert records[0]["page_end"] == 3


@patch("blue_doc_source.source.document.pipeline.PDFParser")
def test_ingest_pdf_upserts_to_vector_source(mock_parser_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_parser_cls.return_value.parse_pages.return_value = [
        {"page_num": 1, "text": "Some content to embed and store."},
    ]
    vector_source = MagicMock()
    pipeline = DocumentPipeline(
        chunk_size=200,
        overlap=0,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    records = pipeline.ingest(str(pdf_path))
    assert vector_source.upsert.call_count == len(records)


@patch("blue_doc_source.source.document.pipeline.PDFParser")
def test_ingest_pdf_section_title(mock_parser_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_parser_cls.return_value.parse_pages.return_value = [
        {"page_num": 1, "text": "# Eligibility\nCustomers must be 18 or older."},
    ]
    pipeline = DocumentPipeline(chunk_size=200, overlap=0)
    records = pipeline.ingest(str(pdf_path))
    assert records[0].get("section_title") == "Eligibility"


def test_stale_cleanup_no_op_without_vector_source(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("first version", encoding="utf-8")
    pipeline = DocumentPipeline(chunk_size=50, overlap=0)
    pipeline.ingest(str(f))
    f.write_text("second version with changed content", encoding="utf-8")
    # should not raise even without a vector_source
    records = pipeline.ingest(str(f))
    assert len(records) > 0


def test_stale_cleanup_delete_failure_does_not_block_ingest(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("first version content here", encoding="utf-8")
    vector_source = MagicMock()
    vector_source.delete.side_effect = Exception("Redis connection lost")
    pipeline = DocumentPipeline(
        chunk_size=50,
        overlap=0,
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )
    pipeline.ingest(str(f))
    f.write_text("completely different second version", encoding="utf-8")
    # delete will fail but ingest should still succeed
    records = pipeline.ingest(str(f))
    assert len(records) > 0
    vector_source.delete.assert_called()


def test_custom_chunker_is_used(tmp_path):
    path = make_txt(tmp_path, " ".join(["word"] * 20))
    window = WindowChunker(chunk_size=5, overlap=1)
    pipeline = DocumentPipeline(chunker=window)
    records = pipeline.ingest(path)
    assert len(records) > 0
