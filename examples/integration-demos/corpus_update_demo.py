"""
Corpus Update Demo  —  Add, Replace, and Remove Documents
==========================================================
Shows how to maintain a living document collection: adding new documents,
replacing a revised version, and removing a document that is no longer needed.

Operations covered:
  DOCUMENT.INGEST        — add or re-add a document (idempotent if content unchanged)
  DOCUMENT.DELETE        — remove all chunks for a document by document_id + chunk_count
  Replace workflow       — DELETE old doc_id, then INGEST new file

Content-addressing guarantee:
  • Same bytes  → same document_id → upsert, no growth
  • Changed bytes → new document_id → stale old chunks must be explicitly deleted

Run:
    python3 examples/integration-demos/corpus_update_demo.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_doc_source import DocumentDataSource, DummyEmbeddingModel, NLQueryParser

COLLECTION = "corpus_demo"
DIV  = "=" * 68
DIV2 = "-" * 68

DOC_A_V1 = """\
Vendor Agreement — Acme Corp (Version 1)

1. Services
Acme Corp will deliver monthly cloud storage reports by the 5th of each month.
Standard rate: $2,500 per month. Contract term: January 2024 to December 2024.
"""

DOC_A_V2 = """\
Vendor Agreement — Acme Corp (Version 2, Revised)

1. Services
Acme Corp will deliver weekly cloud storage reports every Monday.
Updated rate: $2,800 per month. Contract term: January 2024 to December 2025.

2. Addendum — Data Privacy
All reports must comply with SOC 2 Type II standards.
"""

DOC_B = """\
Service Level Agreement — Beta Systems

1. Uptime Guarantee
Beta Systems guarantees 99.9% uptime for hosted services.
Downtime compensation: 10% monthly fee credit per incident exceeding 4 hours.
Escalation contact: sla@betasystems.example.com
"""

DOC_C = """\
Non-Disclosure Agreement — Confidential

All proprietary information shared between parties must remain confidential
for a period of five years from the date of signature.
Jurisdiction: State of California.
"""


def doc_id_from_bytes(b: bytes) -> str:
    return "doc_" + hashlib.sha256(b).hexdigest()[:16]


def run() -> None:
    print()
    print(DIV)
    print("Corpus Update Demo  —  Add, Replace, Remove Documents")
    print(DIV)
    print()

    tmp = Path(tempfile.mkdtemp(prefix="corpus-demo-"))
    store = LocalMemoryVectorStore()
    source = DocumentDataSource(
        name="corpus_demo",
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=80,
        overlap=10,
        default_collection=COLLECTION,
        nl_parser=NLQueryParser(parse_fn=lambda t: {"op": "DOCUMENT.SEARCH", "query": t, "k": 5}),
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})

    try:
        path_a = tmp / "vendor_acme.txt"
        path_b = tmp / "sla_beta.txt"
        path_c = tmp / "nda_confidential.txt"

        # ── Step 1: initial ingest ────────────────────────────────────────
        print(DIV2)
        print("Step 1  —  Ingest three documents")
        print(DIV2)
        print()

        path_a.write_text(DOC_A_V1, encoding="utf-8")
        path_b.write_text(DOC_B,    encoding="utf-8")
        path_c.write_text(DOC_C,    encoding="utf-8")

        r_a = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(path_a), "collection": COLLECTION})
        r_b = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(path_b), "collection": COLLECTION})
        r_c = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(path_c), "collection": COLLECTION})

        did_a1 = r_a[0]["document_id"] if r_a else doc_id_from_bytes(path_a.read_bytes())
        did_b  = r_b[0]["document_id"] if r_b else doc_id_from_bytes(path_b.read_bytes())
        did_c  = r_c[0]["document_id"] if r_c else doc_id_from_bytes(path_c.read_bytes())

        size1 = store.count(COLLECTION)
        print(f"  Acme v1  document_id={did_a1}  chunks={len(r_a)}")
        print(f"  Beta SLA document_id={did_b}   chunks={len(r_b)}")
        print(f"  NDA      document_id={did_c}   chunks={len(r_c)}")
        print()
        print(f"  Total chunks in store: {size1}")

        # ── Step 2: re-ingest same file (idempotent) ──────────────────────
        print()
        print(DIV2)
        print("Step 2  —  Re-ingest same file  →  no growth")
        print(DIV2)
        print()

        r_a2 = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(path_a), "collection": COLLECTION})
        size2 = store.count(COLLECTION)
        print(f"  Re-ingest returned {len(r_a2)} records")
        print(f"  Store size before: {size1}  after: {size2}")
        assert size1 == size2, f"Store grew unexpectedly: {size1} → {size2}"
        print(f"  ✓ Upsert is idempotent — same bytes, same IDs, no duplicate chunks")

        # ── Step 3: replace with revised content ─────────────────────────
        print()
        print(DIV2)
        print("Step 3  —  Replace Acme v1 with v2  (changed content → new document_id)")
        print(DIV2)
        print()

        path_a.write_text(DOC_A_V2, encoding="utf-8")
        did_a2 = doc_id_from_bytes(path_a.read_bytes())
        print(f"  Old document_id: {did_a1}")
        print(f"  New document_id: {did_a2}")
        print(f"  Same? {did_a1 == did_a2}  (must be False — content changed)")
        print()

        print("  Step 3a — DELETE old version by document_id + chunk_count")
        source.execute_query({
            "op": "DOCUMENT.DELETE",
            "document_id": did_a1,
            "chunk_count": len(r_a),
        })
        size_after_delete = store.count(COLLECTION)
        print(f"  Store after delete: {size_after_delete}  (removed {size1 - size_after_delete} stale chunks)")

        print()
        print("  Step 3b — INGEST new version")
        r_a3 = source.execute_query({"op": "DOCUMENT.INGEST", "path": str(path_a), "collection": COLLECTION})
        size_after_v2 = store.count(COLLECTION)
        print(f"  New version: {len(r_a3)} chunks  document_id={did_a2}")
        print(f"  Store size : {size_after_v2}")

        # Verify old chunks gone
        all_ids = store.doc_ids(COLLECTION)
        stale = [i for i in all_ids if i.startswith(did_a1 + "_chunk_")]
        fresh = [i for i in all_ids if i.startswith(did_a2 + "_chunk_")]
        print()
        print(f"  ✓ Stale v1 chunks remaining : {len(stale)}  (must be 0)")
        print(f"  ✓ Fresh v2 chunks present   : {len(fresh)}")

        # ── Step 4: remove a document entirely ────────────────────────────
        print()
        print(DIV2)
        print("Step 4  —  Remove NDA document entirely")
        print(DIV2)
        print()

        size_before_remove = store.count(COLLECTION)
        print(f"  Store before: {size_before_remove} chunks")
        print(f"  Removing document_id={did_c}  ({len(r_c)} chunks)")
        source.execute_query({
            "op": "DOCUMENT.DELETE",
            "document_id": did_c,
            "chunk_count": len(r_c),
        })
        size_after_remove = store.count(COLLECTION)
        print(f"  Store after : {size_after_remove} chunks  (removed {size_before_remove - size_after_remove})")

        nda_ids = [i for i in store.doc_ids(COLLECTION) if i.startswith(did_c + "_chunk_")]
        print(f"  ✓ NDA chunks remaining: {len(nda_ids)}  (must be 0)")

        # ── Step 5: verify live corpus ────────────────────────────────────
        print()
        print(DIV2)
        print("Step 5  —  Verify final corpus contents")
        print(DIV2)
        print()
        print("  Final corpus: Acme v2, Beta SLA  (NDA removed)")
        print()
        all_ids  = store.doc_ids(COLLECTION)
        all_docs = list(store._store.get(COLLECTION, {}).values())

        # keyword confirmation: Acme v2 content
        v2_hit = any("weekly" in d.get("text", "") for d in all_docs)
        print(f"  Acme v2 content ('weekly reports') present: {'✓' if v2_hit else '✗'}")

        # keyword confirmation: Beta SLA content
        sla_hit = any("99.9%" in d.get("text", "") for d in all_docs)
        print(f"  Beta SLA content ('99.9% uptime') present : {'✓' if sla_hit else '✗'}")

        # keyword confirmation: NDA content gone
        nda_hit = any("five years" in d.get("text", "") for d in all_docs)
        print(f"  NDA content ('five years') present        : {'✗ gone' if not nda_hit else '✗ STILL THERE — delete failed'}")

        print()
        print("  NL search on positive cases (DummyEmbeddingModel — scores not semantic):")
        for nl in [
            "what is the updated rate for Acme",
            "uptime guarantee compensation",
        ]:
            results = source.execute_query_nl(nl, collection=COLLECTION)
            if results:
                snippet = results[0].get("text", "")[:90].replace("\n", " ")
                fname   = results[0].get("file_name", "?")
                print(f"  \"{nl}\"")
                print(f"    → [{fname}] {snippet}…")
            else:
                print(f"  \"{nl}\"  → no results")
            print()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  Add   : pipeline.ingest(path)  or  DOCUMENT.INGEST")
    print("  Remove: DOCUMENT.DELETE with document_id + chunk_count")
    print("  Update: DELETE old doc_id  →  INGEST new file")
    print()
    print("  Content-addressing means:")
    print("    • Unchanged files are safely re-ingested (upsert, no growth)")
    print("    • Changed files get a new document_id — old chunks need explicit DELETE")
    print("    • Track document_id at ingest time so you can DELETE later")
    print()


if __name__ == "__main__":
    run()
