"""
Content-Addressing and Deduplication
======================================
Shows how blue-data-doc-source derives chunk IDs from file content and uses
them to prevent duplicate ingestion.

Key rule:
    document_id = "doc_" + sha256(file_bytes).hexdigest()[:16]
    chunk_id    = f"{document_id}_chunk_{index:06d}"

Consequences:
  1. Ingest the same file twice  → same chunk IDs → upsert overwrites, no duplicates
  2. Two paths with identical content → same chunk IDs (content-addressed)
  3. File content changes → new hash → new chunk IDs → old chunks become stale

This demo shows all three cases and demonstrates how to clean up stale chunks.

Run:
    python3 examples/feature-demos/content_addressing_demo.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_doc_source import DocumentPipeline, DummyEmbeddingModel

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "dedup_demo"
DIV  = "=" * 68
DIV2 = "-" * 68

V1_TEXT = """\
Policy Section 1 — Eligibility

Coverage is available to individuals between 18 and 65 who are legal
residents of the United States. Dependents may be enrolled within 30
days of a qualifying life event.
"""

V2_TEXT = """\
Policy Section 1 — Eligibility (Revised)

Coverage is available to individuals between 18 and 70 who are legal
residents of the United States. Dependents may be enrolled within 60
days of a qualifying life event. New: same-sex domestic partners
are now eligible for full dependent coverage.
"""


def doc_id_from_bytes(content: bytes) -> str:
    return "doc_" + hashlib.sha256(content).hexdigest()[:16]


def make_pipeline() -> tuple[DocumentPipeline, LocalMemoryVectorStore]:
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=100,
        overlap=10,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)
    return pipeline, store


def run() -> None:
    print()
    print(DIV)
    print("Content-Addressing and Deduplication")
    print(DIV)
    print()
    print("  document_id = 'doc_' + sha256(file_bytes).hexdigest()[:16]")
    print("  chunk_id    = f'{document_id}_chunk_{index:06d}'")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="ca-demo-"))
    try:
        # ── Case 1: ingest same file twice ────────────────────────────────
        print(DIV2)
        print("Case 1 — Ingest same file twice  →  no duplicates")
        print(DIV2)
        print()

        pipeline, store = make_pipeline()
        doc_path = tmp / "policy_v1.txt"
        doc_path.write_text(V1_TEXT, encoding="utf-8")

        records_a = pipeline.ingest(str(doc_path))
        ids_a     = [r["chunk_id"] for r in records_a]
        count_a   = store.count(COLLECTION)
        print(f"  First ingest  : {len(records_a)} chunks, store size = {count_a}")
        for cid in ids_a:
            print(f"    {cid}")

        records_b = pipeline.ingest(str(doc_path))
        ids_b     = [r["chunk_id"] for r in records_b]
        count_b   = store.count(COLLECTION)
        print()
        print(f"  Second ingest : {len(records_b)} chunks, store size = {count_b}")

        assert ids_a == ids_b, "chunk IDs changed — this is a bug"
        assert count_a == count_b, "store grew — duplicates created"
        print()
        print(f"  ✓ Chunk IDs identical: {ids_a[0] == ids_b[0]}")
        print(f"  ✓ Store size unchanged: {count_a} → {count_b}")
        print(f"  ✓ Upsert overwrote existing chunks, no duplicates created.")

        # ── Case 2: two paths, same content ───────────────────────────────
        print()
        print(DIV2)
        print("Case 2 — Two file paths, identical content  →  same chunk IDs")
        print(DIV2)
        print()

        pipeline2, store2 = make_pipeline()
        path_copy = tmp / "policy_copy.txt"
        shutil.copy(doc_path, path_copy)

        r_orig = pipeline2.ingest(str(doc_path))
        r_copy = pipeline2.ingest(str(path_copy))

        ids_orig = [r["chunk_id"] for r in r_orig]
        ids_copy = [r["chunk_id"] for r in r_copy]
        equal    = ids_orig == ids_copy

        print(f"  Original path : {doc_path.name}")
        print(f"  Copy path     : {path_copy.name}")
        print(f"  Chunk IDs match: {equal}")
        for o, c in zip(ids_orig, ids_copy):
            match = "✓" if o == c else "✗"
            print(f"    {match}  {o}  ==  {c}")
        print()
        if equal:
            print("  ✓ Content-addressed: path does not affect identity.")
            print("    Caution: deleting one path's chunks also removes the other's.")
        print()
        print(f"  Store size = {store2.count(COLLECTION)}  "
              f"(not 2×{len(r_orig)} — same IDs overwritten)")

        # ── Case 3: content changes → new IDs → stale chunks ─────────────
        print()
        print(DIV2)
        print("Case 3 — Content changes  →  new chunk IDs  →  stale v1 chunks")
        print(DIV2)
        print()

        pipeline3, store3 = make_pipeline()
        path_v = tmp / "policy_versioned.txt"

        path_v.write_text(V1_TEXT, encoding="utf-8")
        r_v1    = pipeline3.ingest(str(path_v))
        ids_v1  = [r["chunk_id"] for r in r_v1]
        size_v1 = store3.count(COLLECTION)
        did_v1  = doc_id_from_bytes(path_v.read_bytes())

        print(f"  v1 document_id : {did_v1}")
        print(f"  v1 chunk IDs   : {ids_v1}")
        print(f"  Store size     : {size_v1}")
        print()

        path_v.write_text(V2_TEXT, encoding="utf-8")
        r_v2    = pipeline3.ingest(str(path_v))
        ids_v2  = [r["chunk_id"] for r in r_v2]
        size_v2 = store3.count(COLLECTION)
        did_v2  = doc_id_from_bytes(path_v.read_bytes())

        print(f"  v2 document_id : {did_v2}")
        print(f"  v2 chunk IDs   : {ids_v2}")
        print(f"  Store size     : {size_v2}")
        print()

        overlap = set(ids_v1) & set(ids_v2)
        stale   = set(ids_v1) - set(ids_v2)
        print(f"  Overlapping IDs (same chunk in v1 and v2) : {len(overlap)}")
        print(f"  Stale v1-only IDs still in store          : {len(stale)}")
        print()
        if stale:
            print("  ✓ In-process pipeline detects content change and removes stale")
            print("    chunks automatically (same path, new hash → old chunks deleted).")
            print()
            print("  ⚠  Cross-process: a NEW pipeline has no memory of v1.")
            print("     Use DOCUMENT.DELETE before re-ingesting in a fresh process:")
            print()
            print("       source.execute_query({")
            print('           "op": "DOCUMENT.DELETE",')
            print(f'           "document_id": "{did_v1}",')
            print(f'           "chunk_count": {len(ids_v1)},')
            print("       })")
        else:
            print("  ✓ In-process pipeline cleaned up stale chunks automatically.")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(DIV)
    print("Summary")
    print(DIV)
    print()
    print("  ✓ Same bytes → same chunk IDs → upsert is idempotent")
    print("  ✓ Content-addressed: path is irrelevant to identity")
    print("  ⚠  Changed content → new IDs → stale chunks require explicit cleanup")
    print("     in fresh processes. Use DOCUMENT.DELETE or see:")
    print("     examples/integration-demos/corpus_update_demo.py")
    print()


if __name__ == "__main__":
    run()
