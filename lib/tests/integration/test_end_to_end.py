"""End-to-end integration tests requiring a local Redis Stack instance.

Run with:
    docker run -p 6379:6379 redis/redis-stack-server:latest
    .venv/bin/pytest lib/tests/integration/ -v
"""
import pytest

from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
from blue_data_doc_source.source.document.document_source import DocumentDataSource
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel

from .conftest import requires_redis

COLLECTION = "test_integration_docs"


@pytest.fixture
def redis_source():
    return RedisVectorSource(
        name="test_redis",
        properties={"connection": {"host": "localhost", "port": 6379, "scan_prefix": "test_doc"}},
    )


@pytest.fixture
def source(redis_source):
    # Drop any pre-existing index so each run recreates it with the current
    # schema — index-field changes (e.g. #71's text/chunk_id) require recreation.
    import redis as _redis
    try:
        _redis.Redis(host="localhost", port=6379).execute_command("FT.DROPINDEX", COLLECTION, "DD")
    except _redis.exceptions.ResponseError:
        pass
    return DocumentDataSource(
        name="test_source",
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=redis_source,
        default_collection=COLLECTION,
    )


@requires_redis
def test_ensure_index_creates_index(source):
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})


@requires_redis
def test_ingest_and_search(source, tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text(
        "# Eligibility\nCustomers must be 18 or older.\n"
        "# Payment\nPayment is due within 30 days.",
        encoding="utf-8",
    )

    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})
    records = source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": str(doc),
        "collection": COLLECTION,
    })

    assert len(records) >= 2
    assert any(r.get("section_title") == "Eligibility" for r in records)
    assert any(r.get("section_title") == "Payment" for r in records)

    results = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "eligibility requirements",
        "collection": COLLECTION,
        "k": 5,
    })
    # Regression (#71): Redis search must return chunk content, not just {id, score}.
    assert isinstance(results, list) and results, "Redis search returned no hits"
    top = results[0]
    assert top.get("text"), "hit is missing chunk text"
    assert top.get("chunk_id"), "hit is missing chunk_id"
    assert isinstance(top.get("score"), (int, float)), "score should be numeric"


@requires_redis
def test_reingest_same_file_no_duplicates(source, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Some content for deduplication testing.", encoding="utf-8")

    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})

    first = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(doc), "collection": COLLECTION})
    second = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(doc), "collection": COLLECTION})

    assert first[0]["document_id"] == second[0]["document_id"]
    assert first[0]["chunk_id"] == second[0]["chunk_id"]


@requires_redis
def test_document_type_prefilter_against_redis(redis_source, tmp_path):
    """Regression (#73): the document_type TAG must filter against a real Redis index.

    Exercises the Redis-specific half (`$.document_type` in `_INDEX_EXTRA_FIELDS`)
    that the in-memory unit tests cannot cover. Drops the index first so the fresh
    schema (with document_type) is applied — existing indexes lack the new field.
    """
    import redis as _redis
    from blue_data_doc_source import KeywordClassifier

    coll = "test_doctype_filter"
    client = _redis.Redis(host="localhost", port=6379)
    try:
        client.execute_command("FT.DROPINDEX", coll, "DD")  # best-effort: fresh schema
    except _redis.exceptions.ResponseError:
        pass  # index didn't exist

    source = DocumentDataSource(
        name="test_doctype_source",
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=redis_source,
        classifier=KeywordClassifier(),
        default_collection=coll,
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": coll})

    contract = tmp_path / "contract.txt"
    contract.write_text(
        "MASTER SERVICES AGREEMENT between Acme Corp and Beta LLC (the parties). "
        "Termination with sixty days notice. Governing law: California. "
        "Indemnification, warranties, confidentiality, force majeure, entire agreement.",
        encoding="utf-8",
    )
    invoice = tmp_path / "invoice.txt"
    invoice.write_text(
        "INVOICE number INV-1001. Bill to: Gamma Inc. Vendor: Blue Solutions. "
        "Total amount due $5,400. Due date 2026-01-31. Payment terms net 30. "
        "Termination of services for non-payment.",
        encoding="utf-8",
    )
    contract_id = source.ingest_text(contract.read_text(), file_name="contract.txt")[0]["document_id"]
    invoice_id = source.ingest_text(invoice.read_text(), file_name="invoice.txt")[0]["document_id"]
    assert contract_id != invoice_id

    def ids(filt):
        query = {"op": "DOCUMENT.SEARCH", "query": "termination", "collection": coll, "k": 10}
        if filt is not None:
            query["filter"] = filt
        # Redis search hits carry {id, score}; the id embeds the chunk's document_id.
        return [h["id"] for h in source.execute_query(query)]

    # No filter → both documents mention "termination", so both should appear.
    unfiltered = ids(None)
    assert any(contract_id in i for i in unfiltered)
    assert any(invoice_id in i for i in unfiltered)

    # document_type pre-filter must restrict results to the matching type only.
    contract_hits = ids({"document_type": "contract"})
    assert contract_hits and all(contract_id in i and invoice_id not in i for i in contract_hits)

    invoice_hits = ids({"document_type": "invoice"})
    assert invoice_hits and all(invoice_id in i and contract_id not in i for i in invoice_hits)
