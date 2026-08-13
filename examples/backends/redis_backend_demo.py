"""
Redis Stack Backend  (requires Redis Stack on localhost:6379)
=============================================================
Shows how to plug RedisVectorSource into the pipeline for a persistent,
production-grade vector store — the rest of the pipeline is unchanged.

Redis Stack vs LocalMemoryVectorStore:
  • Persistent: data survives process restarts
  • Scalable: handles millions of vectors
  • Filterable: metadata field filters on search
  • Requires Redis Stack (redis-stack-server or RedisSearch + RedisJSON modules)

Start Redis Stack:
    docker run -p 6379:6379 redis/redis-stack-server:latest

Run:
    python3 examples/backends/redis_backend_demo.py
    python3 examples/backends/redis_backend_demo.py --host redis.example.com --port 6379
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from blue_data.source.vector.redis_vector_source import RedisVectorSource  # type: ignore
    import redis as _redis
except ImportError:
    print("ERROR: RedisVectorSource or redis-py is not installed.")
    print("  pip install blue-data redis")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data_doc_source import DocumentPipeline, DocumentDataSource, DummyEmbeddingModel, NLQueryParser

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "redis_demo"
INDEX_NAME  = f"idx:{COLLECTION}"
DIV  = "=" * 68
DIV2 = "-" * 68


def check_redis(host: str, port: int) -> bool:
    try:
        r = _redis.Redis(host=host, port=port)
        r.ping()
        return True
    except Exception:
        return False


def run(doc_path: str, host: str, port: int) -> None:
    print()
    print(DIV)
    print("Redis Stack Backend Demo")
    print(DIV)
    print()

    if not check_redis(host, port):
        print(f"  ERROR: Cannot connect to Redis at {host}:{port}")
        print()
        print("  Start Redis Stack with:")
        print("    docker run -p 6379:6379 redis/redis-stack-server:latest")
        print()
        print("  Then re-run this demo.")
        return

    print(f"  Redis  : {host}:{port}  ✓ connected")
    print(f"  Doc    : {doc_path}")
    print()

    vector_source = RedisVectorSource(host=host, port=port)

    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )

    # ── Ensure index ────────────────────────────────────────────────────────
    print(DIV2)
    print("Ensure index  —  idempotent (safe to call on every startup)")
    print(DIV2)
    print()
    pipeline.ensure_index(COLLECTION)
    print(f"  ✓ Index ready: {COLLECTION}")

    # ── Ingest ─────────────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Ingest  —  same API as memory backend")
    print(DIV2)
    print()
    records = pipeline.ingest(doc_path)
    print(f"  {len(records)} chunks stored in Redis")
    print(f"  document_id: {records[0]['document_id'] if records else 'n/a'}")

    # ── Re-ingest (idempotent) ─────────────────────────────────────────────
    print()
    print(DIV2)
    print("Re-ingest  —  content unchanged, no duplicates")
    print(DIV2)
    print()
    records2 = pipeline.ingest(doc_path)
    print(f"  ✓ {len(records2)} records returned (same IDs, upsert overwrote)")

    # ── Search ─────────────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Vector search  (DummyEmbeddingModel — scores not semantic)")
    print(DIV2)
    print()
    for query in ["prescription drug", "emergency room", "deductible"]:
        results = pipeline.search(query, k=3, collection=COLLECTION)
        print(f"  Query: {query!r}  →  {len(results)} result(s)")
        for r in results[:2]:
            score   = r.get("score", "n/a")
            section = r.get("section_title", "")
            snippet = r.get("text", "")[:80].replace("\n", " ")
            print(f"    score={score}  [{section}]  {snippet}…")
        print()

    # ── Persistence demonstration ──────────────────────────────────────────
    print(DIV2)
    print("Persistence  —  a fresh pipeline instance reads the same index")
    print(DIV2)
    print()

    pipeline2 = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=RedisVectorSource(host=host, port=port),
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )
    pipeline2.ensure_index(COLLECTION)
    results2 = pipeline2.search("deductible", k=2, collection=COLLECTION)
    print(f"  Fresh pipeline instance — search returned {len(results2)} result(s)")
    print("  ✓ Data persisted across pipeline instances (Redis is not in-memory)")

    # ── DocumentDataSource via execute_query ──────────────────────────────
    print()
    print(DIV2)
    print("DocumentDataSource  —  using op-dispatch interface with Redis")
    print(DIV2)
    print()

    source = DocumentDataSource(
        name="redis_demo",
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=RedisVectorSource(host=host, port=port),
        chunk_size=200,
        overlap=30,
        default_collection=COLLECTION,
        nl_parser=NLQueryParser(),
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})

    results3 = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "enrollment period",
        "collection": COLLECTION,
        "k": 3,
    })
    print(f"  DOCUMENT.SEARCH 'enrollment period' → {len(results3)} result(s)")

    nl_results = source.execute_query_nl("who is eligible", collection=COLLECTION)
    print(f"  execute_query_nl 'who is eligible' → {len(nl_results)} result(s)")

    # ── Cleanup ────────────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Cleanup  —  delete ingested document")
    print(DIV2)
    print()
    doc_id      = records[0]["document_id"]
    chunk_count = len(records)
    source.execute_query({
        "op": "DOCUMENT.DELETE",
        "document_id": doc_id,
        "chunk_count": chunk_count,
    })
    print(f"  Deleted {chunk_count} chunks for document_id={doc_id}")
    verify = pipeline.search("deductible", k=3, collection=COLLECTION)
    print(f"  Search after delete: {len(verify)} result(s)  (0 expected)")

    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  Swapping LocalMemoryVectorStore → RedisVectorSource requires")
    print("  changing only the vector_source= argument.  The pipeline and all")
    print("  op-dispatch logic are identical.  Redis adds persistence and scale.")
    print()
    print("  For real semantic quality, also swap DummyEmbeddingModel →")
    print("  OpenAIEmbeddingModel (see openai_embeddings_demo.py).")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redis Stack backend demo")
    parser.add_argument("path",   nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    args = parser.parse_args()
    run(args.path, args.host, args.port)
