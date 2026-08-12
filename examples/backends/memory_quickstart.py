"""Minimal quickstart — ingest a text file and search it.

No Redis. No API key. DummyEmbeddingModel + LocalMemoryVectorStore.
All data lives in-process and is discarded when the script exits.

Run:
    python3 examples/backends/memory_quickstart.py
    python3 examples/backends/memory_quickstart.py --query "prescription drug tiers"
    python3 examples/backends/memory_quickstart.py --query "emergency room" --top-k 5

Note: DummyEmbeddingModel produces SHA-256-hash-based embeddings, not semantic
      embeddings. Search results are returned but scores are not meaningful.
      Replace with OpenAIEmbeddingModel for real similarity — see
      examples/backends/openai_embeddings_demo.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_doc_source import DocumentPipeline, DummyEmbeddingModel

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "quickstart"
DIV = "=" * 60


def run(path: str, query: str, top_k: int) -> None:
    print()
    print(DIV)
    print("blue-data-doc-source — 15-line quickstart")
    print(DIV)
    print()
    print(f"  File      : {path}")
    print(f"  Embedding : DummyEmbeddingModel (dim=8)")
    print(f"  Storage   : LocalMemoryVectorStore (in-process)")
    print()

    # ── Build pipeline ────────────────────────────────────────────────────
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )

    # ── Ingest ────────────────────────────────────────────────────────────
    pipeline.ensure_index(COLLECTION)
    records = pipeline.ingest(path)

    print(f"  {len(records)} chunks ingested into collection '{COLLECTION}'")
    print()

    for r in records[:3]:
        section = f"  [{r['section_title']}]" if r.get("section_title") else ""
        snippet = r["text"][:70].replace("\n", " ")
        print(f"  {r['chunk_id']}{section}")
        print(f"    {snippet}…")
    if len(records) > 3:
        print(f"  … {len(records) - 3} more chunks")
    print()

    # ── Search ────────────────────────────────────────────────────────────
    print(DIV)
    print(f"  Query: \"{query}\"")
    print(DIV)
    print()
    results = pipeline.search(query, collection=COLLECTION, k=top_k)

    if not results:
        print("  No results.")
    else:
        for i, r in enumerate(results, 1):
            section = f"  [{r['section_title']}]" if r.get("section_title") else ""
            score   = float(r.get("score", 0.0))
            snippet = r.get("text", "")[:120].replace("\n", " ")
            print(f"  [{i}] score={score:.4f}{section}")
            print(f"       {snippet}…")
            print()

    print("  Tip: replace DummyEmbeddingModel with OpenAIEmbeddingModel for")
    print("  semantic similarity — see examples/backends/openai_embeddings_demo.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal doc-source quickstart")
    parser.add_argument("path",    nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--query", default="deductible and out-of-pocket maximum")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run(args.path, args.query, args.top_k)
