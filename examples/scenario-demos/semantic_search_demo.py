"""
Demo 3 — Semantic Search vs Keyword Search  (requires OpenAI)
==============================================================
Problem : Keyword search misses synonyms and paraphrases.  Searching for
          "copay" does not find chunks that say "cost-sharing" or "out-of-pocket".

Solution: OpenAI embeddings map both phrases to nearby vectors, so a cosine
          similarity search returns semantically related chunks even when the
          exact query words are absent.

This demo:
  1. Ingests sample_policy.txt with OpenAIEmbeddingModel
  2. Runs the same queries as keyword search
  3. Shows which results overlap and which are exclusively semantic

Requirements:
  OPENAI_API_KEY environment variable must be set.

Run:
    OPENAI_API_KEY=sk-... python3 examples/scenario-demos/semantic_search_demo.py
    OPENAI_API_KEY=sk-... python3 examples/scenario-demos/semantic_search_demo.py --k 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_doc_source import DocumentPipeline, OpenAIEmbeddingModel

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY is not set.  Export it and retry.")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "semantic_demo"
EMBEDDING_DIM = 1536   # text-embedding-3-small
DIV  = "=" * 68
DIV2 = "-" * 68

QUERY_PAIRS = [
    # (semantic query,          keyword variant)
    ("cost sharing responsibility",       "copay"),
    ("who can join the plan",             "eligibility"),
    ("drugs that are not covered",        "exclusion"),
    ("how much you pay before insurance", "deductible"),
]


def keyword_hits(records: list[dict], word: str, k: int) -> list[dict]:
    return [r for r in records if word.lower() in r.get("text", "").lower()][:k]


def run(doc_path: str, k: int) -> None:
    print()
    print(DIV)
    print("Demo 3 — Semantic Search vs Keyword Search")
    print(DIV)
    print()
    print(f"  Document  : {doc_path}")
    print(f"  Embedding : text-embedding-3-small  (dim={EMBEDDING_DIM})")
    print(f"  k         : {k} results per query")
    print()

    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=OpenAIEmbeddingModel(
            api_key=API_KEY,
            model="text-embedding-3-small",
            dim=EMBEDDING_DIM,
        ),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)

    print("  Ingesting (calling OpenAI for embeddings) …")
    records = pipeline.ingest(doc_path)
    print(f"  {len(records)} chunks embedded and stored\n")

    for sem_query, kw_query in QUERY_PAIRS:
        print(DIV2)
        print(f"Semantic : {sem_query!r}")
        print(f"Keyword  : {kw_query!r}")
        print(DIV2)
        print()

        # Semantic results via vector similarity
        sem_results = pipeline.search(sem_query, k=k, collection=COLLECTION)
        sem_ids     = {r.get("chunk_id") for r in sem_results}

        # Keyword results via substring match
        kw_results  = keyword_hits(records, kw_query, k)
        kw_ids      = {r.get("chunk_id") for r in kw_results}

        overlap     = sem_ids & kw_ids
        sem_only    = sem_ids - kw_ids
        kw_only     = kw_ids - sem_ids

        print(f"  Semantic hits : {len(sem_results)}")
        print(f"  Keyword hits  : {len(kw_results)}")
        print(f"  Overlap       : {len(overlap)}")
        print(f"  Semantic-only : {len(sem_only)}  ← found without exact keyword")
        print(f"  Keyword-only  : {len(kw_only)}  ← exact match missed semantically")
        print()

        print("  Top semantic result:")
        if sem_results:
            r       = sem_results[0]
            snippet = r.get("text", "")[:200].replace("\n", " ")
            section = f"  [{r['section_title']}]" if r.get("section_title") else ""
            score   = r.get("score", "n/a")
            print(f"    score={score:.4f}{section}")
            print(f"    {snippet}…")
        print()

        if sem_only:
            print("  Chunks found only via semantic search (no keyword overlap):")
            for cid in list(sem_only)[:2]:
                r = next((x for x in sem_results if x.get("chunk_id") == cid), None)
                if r:
                    snippet = r.get("text", "")[:160].replace("\n", " ")
                    print(f"    {snippet}…")
        print()

    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  Semantic search finds topically relevant chunks even when the exact")
    print("  query words are absent.  Keyword search is faster but limited to")
    print("  exact substring matches.")
    print()
    print("  Use OpenAIEmbeddingModel for production retrieval quality.")
    print("  DummyEmbeddingModel (SHA-256 hashes) is for integration testing only")
    print("  — its cosine scores are meaningless for semantic retrieval.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic vs keyword search demo")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--k",  type=int,  default=3)
    args = parser.parse_args()
    run(args.path, args.k)
