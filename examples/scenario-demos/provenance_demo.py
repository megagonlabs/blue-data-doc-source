"""
Demo 2 — Every Search Result Knows Where It Came From
=======================================================
Problem : A naive RAG pipeline returns a text snippet. The user asks
          "where does this come from?" and there is no answer.

Solution: blue-data-doc-source attaches provenance to every chunk at ingest
          time.  Each result carries:
            • file_name    — the source document
            • page_start   — page number (PDF only, 1-based)
            • page_end     — always == page_start; chunks never cross pages
            • section_title — the section header the chunk belongs to
            • document_id  — stable content-addressed identifier
            • chunk_id     — position within the document

This demo ingests a PDF, shows the provenance fields on every chunk, then runs
keyword searches and displays the full provenance chain for each result.

Run:
    python3 examples/scenario-demos/provenance_demo.py
    python3 examples/scenario-demos/provenance_demo.py data/sample_policy.pdf
    python3 examples/scenario-demos/provenance_demo.py --format txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_doc_source import DocumentPipeline, DummyEmbeddingModel

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_PDF = DATA_DIR / "sample_policy.pdf"
DEFAULT_TXT = DATA_DIR / "sample_policy.txt"
COLLECTION  = "provenance_demo"
DIV  = "=" * 68
DIV2 = "-" * 68

SEARCH_TERMS = [
    "prescription drug",
    "emergency room",
    "deductible",
    "enrollment",
]


def provenance_line(r: dict) -> str:
    parts = [r.get("file_name", "?")]
    if r.get("page_start") is not None:
        parts.append(f"page {r['page_start']}")
    if r.get("section_title"):
        parts.append(f"§ {r['section_title']}")
    return " › ".join(parts)


def run(doc_path: str) -> None:
    path = Path(doc_path)
    is_pdf = path.suffix.lower() == ".pdf"

    print()
    print(DIV)
    print("Demo 2 — Every Search Result Knows Where It Came From")
    print(DIV)
    print()
    print(f"  Document: {doc_path}")
    print(f"  Format  : {'PDF (page-scoped chunks)' if is_pdf else 'TXT (section-scoped chunks)'}")
    print()

    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=200,
        overlap=30,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)

    print("  Ingesting …")
    records = pipeline.ingest(str(path))
    print(f"  {len(records)} chunks stored\n")

    # ── Provenance inventory ───────────────────────────────────────────────
    print(DIV2)
    print("Provenance fields on every chunk")
    print(DIV2)
    print()
    print(f"  {'chunk_id':<42}  {'file_name':<22}  {'page':>4}  {'section_title'}")
    print(f"  {'-'*42}  {'-'*22}  {'-'*4}  {'-'*30}")
    for r in records[:12]:
        cid     = r.get("chunk_id", "")[:42]
        fname   = r.get("file_name", "")[:22]
        page    = str(r["page_start"]) if r.get("page_start") is not None else "—"
        section = (r.get("section_title") or "")[:40]
        print(f"  {cid:<42}  {fname:<22}  {page:>4}  {section}")
    if len(records) > 12:
        print(f"  … {len(records) - 12} more chunks")

    print()
    if is_pdf:
        print("  PDF chunks: page_start == page_end (chunks never cross page boundaries)")
    else:
        print("  TXT chunks: page_start/page_end are None; section_title set from headers")

    # ── Field reference ────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Full field reference (first chunk)")
    print(DIV2)
    print()
    r0 = records[0]
    for field in [
        "document_id", "chunk_id", "file_name", "source_type", "mime_type",
        "page_start", "page_end", "section_title",
    ]:
        present = field in r0
        value   = r0.get(field, "—  (omitted when None)")
        print(f"  {field:<20}  {value!r:>40}  {'✓' if present else '(absent)'}")

    print()
    print("  Note: ChunkRecord.to_dict() omits None optional fields entirely.")
    print("  Check key presence, not value: 'page_start' in chunk, not chunk['page_start'] is not None")

    # ── Keyword search with provenance ────────────────────────────────────
    print()
    print(DIV2)
    print("Keyword search  →  full provenance chain per result")
    print(DIV2)

    for term in SEARCH_TERMS:
        hits = [r for r in records if term.lower() in r.get("text", "").lower()]
        print()
        print(f"  Query: \"{term}\"  →  {len(hits)} matching chunk(s)")
        for r in hits[:3]:
            prov    = provenance_line(r)
            snippet = r.get("text", "")[:120].replace("\n", " ")
            print(f"    Source  : {prov}")
            print(f"    Chunk   : {r.get('chunk_id')}")
            print(f"    Snippet : {snippet}…")
            print()

    # ── RAG citation example ───────────────────────────────────────────────
    print(DIV2)
    print("RAG citation  —  how an agent would format a cited answer")
    print(DIV2)
    print()
    term = "prescription drug"
    hits = [r for r in records if term in r.get("text", "").lower()]
    if hits:
        r = hits[0]
        print("  Question: What are the prescription drug copay tiers?")
        print()
        print("  Answer (grounded):")
        print(f"    {r.get('text', '')[:300].replace(chr(10), ' ')}…")
        print()
        print(f"  Source: {provenance_line(r)}")
        print(f"  Chunk : {r.get('chunk_id')}")

    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  Every chunk carries a complete provenance chain:")
    print("    document_id → file_name → page → section_title → chunk position")
    print()
    print("  Agents can cite exactly which page of which document supports")
    print("  each claim — without any post-hoc retrieval of metadata.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provenance demo")
    parser.add_argument("path",     nargs="?", default=str(DEFAULT_PDF))
    parser.add_argument("--format", choices=["pdf", "txt"],
                        help="override to txt if PDF parsing unavailable")
    args = parser.parse_args()

    doc = args.path
    if args.format == "txt":
        doc = str(DEFAULT_TXT)
    run(doc)
