import pytest
from unittest.mock import MagicMock

from blue_data_doc_source.source.document.document_source import DocumentDataSource
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel
from blue_data_doc_source.source.document.enrichment import BasicEnricher


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


# --- OCR wiring ---

def test_document_source_ocr_fallback_wired_to_pipeline():
    source = DocumentDataSource(ocr_fallback=True)
    assert source._pipeline.pdf_parser._ocr_fallback is True


def test_document_source_ocr_fallback_defaults_to_false():
    source = DocumentDataSource()
    assert source._pipeline.pdf_parser._ocr_fallback is False


# --- DOCUMENT.FIND_THEN_SEARCH ---

def make_doc(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return str(f)


def test_find_then_search_returns_chunks_from_matched_doc(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    invoice_content = (
        "invoice number INV-001 due date amount due bill to line item payment "
        "subtotal tax total remit vendor quantity unit price invoice number "
    ) * 4
    path = make_doc(tmp_path, "inv.txt", invoice_content)
    source = DocumentDataSource(
        chunk_size=20, overlap=2,
        classifier=KeywordClassifier(),
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    results = source.execute_query({
        "op": "DOCUMENT.FIND_THEN_SEARCH",
        "find_filter": {"document_type": "invoice"},
        "query": "payment due",
        "k": 3,
        "mode": "bm25",
    })
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert "text" in r
        assert "score" in r


def test_find_then_search_empty_when_no_match(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    path = make_doc(tmp_path, "doc.txt", " ".join(["invoice"] * 30))
    source = DocumentDataSource(
        chunk_size=20, overlap=2,
        classifier=KeywordClassifier(),
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    results = source.execute_query({
        "op": "DOCUMENT.FIND_THEN_SEARCH",
        "find_filter": {"document_type": "contract"},
        "query": "termination",
        "k": 3,
        "mode": "bm25",
    })
    assert results == []


def test_find_then_search_searches_multiple_matched_docs(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    base = "invoice number due date amount due bill to line item payment subtotal tax total remit vendor quantity unit price "
    inv1 = make_doc(tmp_path, "inv1.txt", (base + "alpha payment alpha ") * 4)
    inv2 = make_doc(tmp_path, "inv2.txt", (base + "beta payment beta ") * 4)
    source = DocumentDataSource(
        chunk_size=20, overlap=2,
        classifier=KeywordClassifier(),
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": inv1})
    source.execute_query({"op": "DOCUMENT.INGEST", "path": inv2})
    results = source.execute_query({
        "op": "DOCUMENT.FIND_THEN_SEARCH",
        "find_filter": {"document_type": "invoice"},
        "query": "payment",
        "k": 10,
        "mode": "bm25",
    })
    doc_ids = {r["document_id"] for r in results}
    assert len(doc_ids) == 2


def test_find_then_search_missing_query_raises():
    source = DocumentDataSource()
    with pytest.raises(ValueError, match="query"):
        source.execute_query({
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": "invoice"},
        })


def test_find_then_search_results_sorted_by_score(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    invoice_content = (
        "invoice number INV-001 due date amount due bill to line item payment "
        "subtotal tax total remit vendor quantity unit price invoice number "
    ) * 4
    path = make_doc(tmp_path, "inv.txt", invoice_content)
    source = DocumentDataSource(
        chunk_size=20, overlap=2,
        classifier=KeywordClassifier(),
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    results = source.execute_query({
        "op": "DOCUMENT.FIND_THEN_SEARCH",
        "find_filter": {"document_type": "invoice"},
        "query": "payment",
        "k": 5,
        "mode": "bm25",
    })
    scores = [r.get("score") or 0 for r in results]
    assert scores == sorted(scores, reverse=True)
