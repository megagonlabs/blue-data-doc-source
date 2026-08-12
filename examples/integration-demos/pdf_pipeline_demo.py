"""
PDF Pipeline Demo
=================
Shows what changes when you ingest a PDF instead of a plain-text file.

Key differences vs TXT ingestion:
  • PDFParser splits text page-by-page (uses pdfplumber under the hood)
  • Each chunk carries page_start == page_end  (chunks never span pages)
  • section_title comes from the page-level heading detection, not file sections
  • document_id is still SHA-256 of the raw PDF bytes
  • SectionChunker applied per page, WindowChunker available as override

Run:
    python3 examples/integration-demos/pdf_pipeline_demo.py
    python3 examples/integration-demos/pdf_pipeline_demo.py path/to/file.pdf
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
COLLECTION  = "pdf_demo"
DIV  = "=" * 68
DIV2 = "-" * 68


def ingest_and_show(pipeline: DocumentPipeline, path: str, label: str) -> list[dict]:
    print(f"\n  Ingesting {label}: {Path(path).name}")
    records = pipeline.ingest(path)
    print(f"  → {len(records)} chunks")
    return records


def show_page_distribution(records: list[dict]) -> None:
    pages: dict[int | None, int] = {}
    for r in records:
        p = r.get("page_start")
        pages[p] = pages.get(p, 0) + 1

    if all(p is None for p in pages):
        print("  (page_start is None for all chunks — TXT file)")
        return

    print(f"  {'Page':<6}  {'Chunks':>6}")
    print(f"  {'-'*6}  {'-'*6}")
    for page in sorted(p for p in pages if p is not None):
        print(f"  {page:<6}  {pages[page]:>6}")


def run(pdf_path: str) -> None:
    print()
    print(DIV)
    print("PDF Pipeline Demo")
    print(DIV)
    print()

    # ── PDF ingest ─────────────────────────────────────────────────────────
    print(DIV2)
    print("PDF ingest  —  page_start, page_end, section_title per chunk")
    print(DIV2)

    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=150,
        overlap=20,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)

    if not Path(pdf_path).exists():
        print(f"\n  PDF not found: {pdf_path}")
        print("  Tip: supply a PDF path as an argument, or create one from the TXT:")
        print("       python3 -c \"import fpdf; ...\"  (see README)")
        print()
        print("  Falling back to TXT to show the structural difference ...")
        _run_txt_comparison(store, pipeline)
        return

    records = ingest_and_show(pipeline, pdf_path, "PDF")
    print()

    # ── Page distribution ──────────────────────────────────────────────────
    print(DIV2)
    print("Chunk distribution across pages")
    print(DIV2)
    print()
    show_page_distribution(records)

    # ── Per-chunk provenance ───────────────────────────────────────────────
    print()
    print(DIV2)
    print("Provenance fields on first 10 PDF chunks")
    print(DIV2)
    print()
    print(f"  {'chunk_id':<40}  {'page':>4}  {'section_title'}")
    print(f"  {'-'*40}  {'-'*4}  {'-'*30}")
    for r in records[:10]:
        cid     = r.get("chunk_id", "")[:40]
        page    = str(r["page_start"]) if r.get("page_start") is not None else "—"
        section = (r.get("section_title") or "")[:35]
        print(f"  {cid:<40}  {page:>4}  {section}")

    # ── Chunk-never-spans-pages assertion ─────────────────────────────────
    print()
    print(DIV2)
    print("Verifying chunks never span page boundaries")
    print(DIV2)
    print()
    violations = [r for r in records
                  if r.get("page_start") is not None
                  and r.get("page_end") is not None
                  and r["page_start"] != r["page_end"]]
    if violations:
        print(f"  FAIL: {len(violations)} chunks span multiple pages (unexpected)")
        for v in violations[:3]:
            print(f"    {v['chunk_id']}  page {v['page_start']}–{v['page_end']}")
    else:
        print(f"  ✓ All {len(records)} chunks have page_start == page_end")
        print("  PDF chunks are strictly page-scoped (V1 design constraint).")

    # ── Document ID stability ──────────────────────────────────────────────
    print()
    print(DIV2)
    print("Re-ingest stability  —  same PDF bytes → same chunk IDs")
    print(DIV2)
    print()
    pipeline2 = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=150,
        overlap=20,
        collection=COLLECTION,
    )
    pipeline2.ensure_index(COLLECTION)
    records2 = pipeline2.ingest(pdf_path)

    ids1 = [r["chunk_id"] for r in records]
    ids2 = [r["chunk_id"] for r in records2]
    match = ids1 == ids2
    print(f"  First ingest  : {len(ids1)} chunks")
    print(f"  Second ingest : {len(ids2)} chunks")
    print(f"  IDs identical : {'✓ yes' if match else '✗ no — check hash stability'}")

    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  PDF pipeline adds page-scoped provenance to every chunk.")
    print("  page_start == page_end is a V1 design constraint: chunks never cross pages.")
    print("  Otherwise the pipeline behaves identically to TXT ingestion:")
    print("  same content-addressed IDs, same upsert idempotency, same enrichment.")
    print()


def _run_txt_comparison(
    store: LocalMemoryVectorStore,
    pipeline: DocumentPipeline,
) -> None:
    """Side-by-side TXT vs TXT to show the provenance field difference."""
    txt_path = str(DEFAULT_TXT)
    if not Path(txt_path).exists():
        print("  sample_policy.txt not found either. Exiting.")
        return

    records = ingest_and_show(pipeline, txt_path, "TXT")
    print()
    print(f"  {'chunk_id':<40}  {'page':>8}  {'section_title'}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*30}")
    for r in records[:6]:
        cid     = r.get("chunk_id", "")[:40]
        page    = str(r.get("page_start")) if r.get("page_start") is not None else "None"
        section = (r.get("section_title") or "")[:35]
        print(f"  {cid:<40}  {page:>8}  {section}")

    print()
    print("  TXT: page_start is None, section_title comes from header detection.")
    print("  PDF: page_start == page_end (1-based), section_title from per-page headers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF pipeline demo")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PDF))
    args = parser.parse_args()
    run(args.path)
