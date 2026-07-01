"""End-to-end integration tests requiring a local Redis Stack instance.

Run with:
    docker run -p 6379:6379 redis/redis-stack-server:latest
    .venv/bin/pytest lib/tests/integration/ -v
"""
import pytest

from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
from blue_doc_source.source.document.document_source import DocumentDataSource
from blue_doc_source.source.document.embeddings import DummyEmbeddingModel

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
    assert isinstance(results, list)


@requires_redis
def test_reingest_same_file_no_duplicates(source, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Some content for deduplication testing.", encoding="utf-8")

    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})

    first = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(doc), "collection": COLLECTION})
    second = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(doc), "collection": COLLECTION})

    assert first[0]["document_id"] == second[0]["document_id"]
    assert first[0]["chunk_id"] == second[0]["chunk_id"]
