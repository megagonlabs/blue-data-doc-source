"""
OpenAI Embeddings Backend  (requires OPENAI_API_KEY)
=====================================================
Shows how to swap DummyEmbeddingModel for OpenAIEmbeddingModel to get
real semantic search quality — the only change is the embedding_model argument.

Demonstrates:
  1. Ingest with text-embedding-3-small
  2. Semantic similarity between paraphrases (same meaning, different words)
  3. Comparing retrieval quality vs DummyEmbeddingModel

Run:
    OPENAI_API_KEY=sk-... python3 examples/backends/openai_embeddings_demo.py
    OPENAI_API_KEY=sk-... python3 examples/backends/openai_embeddings_demo.py --model text-embedding-3-large
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel, OpenAIEmbeddingModel

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY is not set.  Export it and retry.")
    sys.exit(1)

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "openai_backend"
DIV  = "=" * 68
DIV2 = "-" * 68

MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

QUERY_PAIRS = [
    # (semantic paraphrase, exact keyword)
    ("what costs do I pay before insurance kicks in",          "deductible"),
    ("who qualifies for coverage and how to sign up",          "eligibility"),
    ("medicine that requires a prescription and its tiers",    "prescription drug"),
    ("how much I pay when I go to the hospital urgently",      "emergency room copay"),
]


def build_pipeline(model: str, dim: int) -> tuple[DocumentPipeline, LocalMemoryVectorStore]:
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=OpenAIEmbeddingModel(api_key=API_KEY, model=model, dim=dim),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)
    return pipeline, store


def build_dummy_pipeline() -> tuple[DocumentPipeline, LocalMemoryVectorStore]:
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)
    return pipeline, store


def run(doc_path: str, model: str) -> None:
    dim = MODEL_DIMS.get(model, 1536)

    print()
    print(DIV)
    print("OpenAI Embeddings Backend")
    print(DIV)
    print()
    print(f"  Document : {doc_path}")
    print(f"  Model    : {model}  (dim={dim})")
    print()

    # ── Build both pipelines ───────────────────────────────────────────────
    print(DIV2)
    print("Ingesting with OpenAIEmbeddingModel …")
    print(DIV2)
    openai_pipeline, openai_store = build_pipeline(model, dim)
    records = openai_pipeline.ingest(doc_path)
    print(f"  {len(records)} chunks stored with OpenAI embeddings")
    print()

    print(DIV2)
    print("Ingesting with DummyEmbeddingModel (baseline — not semantic) …")
    print(DIV2)
    dummy_pipeline, dummy_store = build_dummy_pipeline()
    dummy_pipeline.ingest(doc_path)
    print(f"  {len(records)} chunks stored with dummy SHA-256 embeddings")
    print()

    # ── Compare retrieval quality ──────────────────────────────────────────
    print(DIV2)
    print("Semantic retrieval quality comparison  (k=3 per query)")
    print(DIV2)
    print()
    print("  Paraphrase queries — no exact keyword in the query")
    print()

    for paraphrase, keyword in QUERY_PAIRS:
        openai_hits = openai_pipeline.search(paraphrase, k=3, collection=COLLECTION)
        dummy_hits  = dummy_pipeline.search(paraphrase, k=3, collection=COLLECTION)

        def has_keyword(results: list[dict]) -> int:
            return sum(1 for r in results if keyword.lower() in r.get("text", "").lower())

        openai_kw = has_keyword(openai_hits)
        dummy_kw  = has_keyword(dummy_hits)

        print(f"  Query  : \"{paraphrase}\"")
        print(f"  Target : chunks containing \"{keyword}\"")
        print(f"           OpenAI→ {openai_kw}/3 target chunks retrieved"
              f"   |   Dummy→ {dummy_kw}/3 (random; scores meaningless)")
        if openai_hits:
            top = openai_hits[0]
            snippet = top.get("text", "")[:120].replace("\n", " ")
            score   = top.get("score", "n/a")
            print(f"           Top OpenAI result (score={score:.4f}): {snippet}…")
        print()

    # ── Embedding dimension sanity ─────────────────────────────────────────
    print(DIV2)
    print("Embedding dimensions")
    print(DIV2)
    print()
    print(f"  {model:<30}  dim={dim}")
    print(f"  {'DummyEmbeddingModel':<30}  dim=8  (SHA-256 hash truncated)")
    print()
    print("  ensure_index() creates the vector index with the correct dim.")
    print("  Mixing embedding models in the same index causes silent score errors.")
    print("  Always recreate the index when changing models.")

    # ── Code comparison ────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Swapping models  —  one argument changes")
    print(DIV2)
    print()
    print("  # Development / testing:")
    print("  pipeline = DocumentPipeline(")
    print("      embedding_model=DummyEmbeddingModel(dim=8),")
    print("      vector_source=LocalMemoryVectorStore(),")
    print("  )")
    print()
    print("  # Production:")
    print("  pipeline = DocumentPipeline(")
    print(f"      embedding_model=OpenAIEmbeddingModel(api_key=OPENAI_KEY, model={model!r}, dim={dim}),")
    print("      vector_source=RedisVectorSource(…),")
    print("  )")
    print()

    print(DIV)
    print("Result")
    print(DIV)
    print()
    print(f"  OpenAIEmbeddingModel ({model}) provides real semantic similarity.")
    print("  DummyEmbeddingModel is for tests only — cosine scores are meaningless.")
    print("  Swap the embedding_model argument to upgrade without changing any other code.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI embeddings backend demo")
    parser.add_argument("path",    nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--model", default="text-embedding-3-small",
                        choices=list(MODEL_DIMS.keys()))
    args = parser.parse_args()
    run(args.path, args.model)
