"""
Multi-Format Ingestion Demo
============================
Shows that DocumentPipeline handles .txt, .pdf, and .docx with a single
call to ingest() — format detection is automatic based on file extension.

Key points:
  • TextParser   → .txt files
  • PDFParser    → .pdf files (pdfplumber; page-scoped chunks)
  • DocxParser   → .docx files (python-docx)
  • The pipeline interface is identical for all formats
  • Chunk schema (ChunkRecord) is format-agnostic; only page_start/page_end
    and mime_type differ meaningfully across formats

This demo uses ingest_text() and ingest_pages() bypass methods to simulate
multi-format behaviour without requiring all three file types:
  • ingest_text()  — as if reading a .txt file
  • ingest_pages() — as if reading a multi-page .pdf / .docx

Run:
    python3 examples/integration-demos/multi_format_demo.py
    python3 examples/integration-demos/multi_format_demo.py data/report.txt data/report.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_TXT = DATA_DIR / "sample_policy.txt"
DIV  = "=" * 68
DIV2 = "-" * 68

SAMPLE_PAGES = [
    "Page 1 — Executive Summary\n\nThis health insurance policy covers preventive care, "
    "emergency room visits, and prescription drug costs. Members must meet eligibility "
    "requirements described on page two of this document.",

    "Page 2 — Eligibility Requirements\n\nCoverage is available to full-time employees "
    "aged 18 to 65. Dependents may be added within 30 days of a qualifying life event. "
    "Part-time employees working fewer than 20 hours per week are not eligible.",

    "Page 3 — Coverage and Benefits\n\nIn-network preventive care is covered at 100% with "
    "no deductible. Emergency room visits carry a $300 copayment. Prescription drugs are "
    "covered under a three-tier formulary: generic ($10), brand ($35), specialty ($70).",
]


def make_pipeline() -> tuple[DocumentPipeline, LocalMemoryVectorStore]:
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=120,
        overlap=15,
        collection="multi_format",
    )
    pipeline.ensure_index("multi_format")
    return pipeline, store


def show_chunks(records: list[dict], label: str) -> None:
    print(f"\n  {label}  →  {len(records)} chunk(s)")
    print(f"  {'chunk_id':<42}  {'mime_type':<18}  {'page':>5}  {'section'}")
    print(f"  {'-'*42}  {'-'*18}  {'-'*5}  {'-'*25}")
    for r in records[:8]:
        cid     = r.get("chunk_id", "")[:42]
        mime    = r.get("mime_type", "")[:18]
        page    = str(r["page_start"]) if r.get("page_start") is not None else "—"
        section = (r.get("section_title") or "")[:25]
        print(f"  {cid:<42}  {mime:<18}  {page:>5}  {section}")


def run(txt_path: str | None, pdf_path: str | None) -> None:
    print()
    print(DIV)
    print("Multi-Format Ingestion Demo")
    print(DIV)
    print()
    print("  The same pipeline.ingest() call works for .txt, .pdf, and .docx.")
    print("  Format detection is automatic; the chunk schema is always ChunkRecord.")
    print()

    # ── Method 1: real files ───────────────────────────────────────────────
    if txt_path or pdf_path:
        print(DIV2)
        print("Real files")
        print(DIV2)
        pipeline, store = make_pipeline()
        if txt_path and Path(txt_path).exists():
            records = pipeline.ingest(txt_path)
            show_chunks(records, f"TXT  {Path(txt_path).name}")
        if pdf_path and Path(pdf_path).exists():
            records = pipeline.ingest(pdf_path)
            show_chunks(records, f"PDF  {Path(pdf_path).name}")
        print()

    # ── Method 2: ingest_text() — same as TXT ────────────────────────────
    print(DIV2)
    print("ingest_text()  —  flat text, no page metadata  (like a .txt file)")
    print(DIV2)

    pipeline, store = make_pipeline()
    flat_text = "\n\n".join(SAMPLE_PAGES)
    records_txt = pipeline.ingest_text(
        text=flat_text,
        file_name="policy_report.txt",
        mime_type="text/plain",
    )
    show_chunks(records_txt, "ingest_text()")
    print()

    has_pages = any(r.get("page_start") is not None for r in records_txt)
    print(f"  page_start present: {'yes' if has_pages else 'no (None for all chunks)'}")

    # ── Method 3: ingest_pages() — like a .pdf ───────────────────────────
    print()
    print(DIV2)
    print("ingest_pages()  —  page-scoped chunks  (like a .pdf / .docx)")
    print(DIV2)

    pipeline2, store2 = make_pipeline()
    records_pdf = pipeline2.ingest_pages(
        pages=[{"text": p, "page_num": i + 1} for i, p in enumerate(SAMPLE_PAGES)],
        file_name="policy_report_scanned.pdf",
        mime_type="application/pdf",
    )
    show_chunks(records_pdf, "ingest_pages()")
    print()

    pages_present = [r.get("page_start") for r in records_pdf if r.get("page_start")]
    print(f"  Pages with chunks: {sorted(set(pages_present))}")
    cross = [r for r in records_pdf
             if r.get("page_start") is not None and r.get("page_end") is not None
             and r["page_start"] != r["page_end"]]
    print(f"  Chunks spanning multiple pages: {len(cross)}  (must be 0)")

    # ── Different IDs across formats ───────────────────────────────────────
    print()
    print(DIV2)
    print("Content-addressing across formats")
    print(DIV2)
    print()
    ids_txt = {r["chunk_id"] for r in records_txt}
    ids_pdf = {r["chunk_id"] for r in records_pdf}
    overlap = ids_txt & ids_pdf
    print(f"  TXT chunk count  : {len(records_txt)}")
    print(f"  PDF chunk count  : {len(records_pdf)}")
    print(f"  Shared chunk IDs : {len(overlap)}")
    print()
    print("  IDs differ even for the same text because the document hash covers")
    print("  file_name + mime_type + content — different bytes produce different IDs.")
    print()
    print("  ✓ This is correct: a PDF and a TXT of the same policy are different")
    print("    documents; both can coexist in the same collection without collision.")

    # ── Unified search across formats ─────────────────────────────────────
    print()
    print(DIV2)
    print("Unified search across formats  —  same query hits both corpora")
    print(DIV2)
    print()

    combined_pipeline, combined_store = make_pipeline()
    combined_pipeline.ingest_text(flat_text, file_name="report.txt",
                                  mime_type="text/plain")
    combined_pipeline.ingest_pages(
        [{"text": p, "page_num": i + 1} for i, p in enumerate(SAMPLE_PAGES)],
        file_name="report.pdf",
        mime_type="application/pdf",
    )

    total = combined_store.count("multi_format")
    print(f"  Total chunks in combined store: {total}")
    print()
    print("  (Semantic search requires OpenAI embeddings — see semantic_search_demo.py)")
    print("  Keyword scan across all records:")

    all_records = list(combined_store._store.get("multi_format", {}).values())
    for term in ["eligibility", "prescription drug", "emergency room"]:
        hits = [r for r in all_records if term.lower() in r.get("text", "").lower()]
        sources = sorted({r.get("file_name", "?") for r in hits})
        print(f"    {term!r:<24}  {len(hits)} chunk(s) from: {sources}")

    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  One pipeline, three formats:")
    print("    pipeline.ingest('doc.txt')   → TextParser  → flat chunks, page=None")
    print("    pipeline.ingest('doc.pdf')   → PDFParser   → page-scoped, page=1,2,…")
    print("    pipeline.ingest('doc.docx')  → DocxParser  → page-scoped, page=1,2,…")
    print()
    print("  Bypass methods for programmatic input:")
    print("    pipeline.ingest_text(text, ...)       → flat, like .txt")
    print("    pipeline.ingest_pages(pages, ...)     → page-scoped, like .pdf")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-format ingestion demo")
    parser.add_argument("txt",  nargs="?", default=None, help="Optional .txt file path")
    parser.add_argument("pdf",  nargs="?", default=None, help="Optional .pdf file path")
    args = parser.parse_args()
    run(args.txt, args.pdf)
