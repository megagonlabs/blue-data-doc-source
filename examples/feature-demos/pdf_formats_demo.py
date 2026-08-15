"""PDF formats demo — tables, images, multi-page, real-world, and scanned OCR.

Shows how the pipeline handles five different PDF types:
  1. Invoice with a rendered table (cells extracted as flat text)
  2. Mixed content — text + embedded image + table (image silently skipped)
  3. Multi-page technical report with sections and tables
  4. Real-world research paper (arXiv) — multi-column, figures, references
  5. Scanned board minutes — image-only pages, OCR fallback required

Run:
    python3 examples/feature-demos/pdf_formats_demo.py

No API key needed. OCR requires: pip install blue-data-doc-source[ocr]
and system Tesseract: brew install tesseract poppler
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib" / "src"))

DATA = ROOT / "data"

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel
from blue_data_doc_source.source.document.parsers.pdf_parser import PDFParser


def _hr():
    print("-" * 70)


def _section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _check_data_files():
    needed = [
        "sample_table_invoice.pdf",
        "sample_mixed_content.pdf",
        "sample_multipage_tables.pdf",
        "real_research_paper.pdf",
        "real_annual_report.pdf",
        "sample_scanned_minutes.pdf",
    ]
    missing = [f for f in needed if not (DATA / f).exists()]
    if missing:
        print("Missing data files. Generate them first:")
        print("  python3 data/_generate_realistic_pdfs.py")
        print("  python3 data/_generate_scanned_pdf.py")
        for f in missing:
            print(f"  missing: {f}")
        sys.exit(1)


def step1_table_invoice(pipeline):
    _section("Step 1 — Invoice with table (sample_table_invoice.pdf)")
    print("PDF has a rendered table of line items. pypdf extracts cell text")
    print("as flat prose — table structure (rows/columns) is NOT preserved.\n")

    records = pipeline.ingest(str(DATA / "sample_table_invoice.pdf"))
    print(f"Chunks produced: {len(records)}")
    _hr()
    for r in records:
        words = len(r["text"].split())
        print(f"  chunk {r['chunk_id'][-10:]}  page={r.get('page_start', '-')}  words={words}")
        print(f"  text: {r['text'][:120].strip()!r}")
        print()

    print("Note: table rows appear as flat text, no row/column delimiters.")


def step2_mixed_content(pipeline):
    _section("Step 2 — Mixed text + image + table (sample_mixed_content.pdf)")
    print("PDF embeds a bar chart image alongside text and a table.")
    print("pypdf silently ignores the image — only text is extracted.\n")

    records = pipeline.ingest(str(DATA / "sample_mixed_content.pdf"))
    print(f"Chunks produced: {len(records)}")
    _hr()
    for r in records:
        print(f"  chunk {r['chunk_id'][-10:]}  words={len(r['text'].split())}")
        print(f"  {r['text'][:140].strip()!r}")
        print()

    print("The bar chart (Apr/May/Jun revenue) produced zero text — image skipped.")


def step3_multipage(pipeline):
    _section("Step 3 — Multi-page technical report (sample_multipage_tables.pdf)")
    print("4-page PDF: cover, abstract+body, results table, references.")
    print("Each page becomes a separate parse unit; SectionChunker detects headers.\n")

    records = pipeline.ingest(str(DATA / "sample_multipage_tables.pdf"))
    print(f"Pages parsed: 4   Chunks produced: {len(records)}")
    _hr()
    for r in records:
        section = r.get("section_title", "(no section)")
        print(
            f"  chunk {r['chunk_id'][-10:]}  "
            f"page={r.get('page_start', '-')}  "
            f"section={section!r}  "
            f"words={len(r['text'].split())}"
        )


def step4_real_research(pipeline):
    _section("Step 4 — Real arXiv paper: Attention Is All You Need (real_research_paper.pdf)")
    print("15-page research paper. Multi-column layout, equations, figure captions,")
    print("references. pypdf merges columns into a single text stream per page.\n")

    records = pipeline.ingest(str(DATA / "real_research_paper.pdf"))
    total_words = sum(len(r["text"].split()) for r in records)
    pages = sorted(set(r.get("page_start") for r in records if r.get("page_start")))
    print(f"Pages parsed: {len(pages)}   Chunks produced: {len(records)}   Total words: {total_words:,}")
    _hr()
    print("Sample chunks (first 3):")
    for r in records[:3]:
        print(f"\n  chunk {r['chunk_id'][-10:]}  page={r.get('page_start', '-')}  words={len(r['text'].split())}")
        print(f"  {r['text'][:160].strip()!r}")

    print("\n  ...")
    print(f"\n  chunk {records[-1]['chunk_id'][-10:]}  page={records[-1].get('page_start', '-')}  (last chunk)")
    print(f"  {records[-1]['text'][:160].strip()!r}")


def step5_real_annual_report(pipeline):
    _section("Step 5 — Real annual report: Berkshire Hathaway 2023 (real_annual_report.pdf)")
    print("150-page financial report. Dense tables, footnotes, letter from Warren Buffett.")
    print("Tests pipeline stability on a large real-world document.\n")

    records = pipeline.ingest(str(DATA / "real_annual_report.pdf"))
    total_words = sum(len(r["text"].split()) for r in records)
    pages = sorted(set(r.get("page_start") for r in records if r.get("page_start")))
    print(f"Pages parsed: {len(pages)}   Chunks produced: {len(records)}   Total words: {total_words:,}")
    _hr()
    print("Sample: first chunk (shareholder letter)")
    r = records[0]
    print(f"  page={r.get('page_start', '-')}  words={len(r['text'].split())}")
    print(f"  {r['text'][:200].strip()!r}")

    print("\nSample: search for 'operating earnings'")
    results = pipeline.search("operating earnings per share", k=2)
    for res in results:
        print(f"  page={res.get('page_start', '-')}  score={res.get('score', 0):.4f}")
        print(f"  {res['text'][:120].strip()!r}")


def step6_scanned_ocr():
    _section("Step 6 — Scanned board minutes: OCR fallback (sample_scanned_minutes.pdf)")
    print("Image-only PDF (no text layer) — simulates a document scanned on a photocopier.")
    print("pypdf extracts zero text. OCR fallback runs Tesseract page by page.\n")

    # First show without OCR
    parser_no_ocr = PDFParser(ocr_fallback=False)
    pages_no_ocr = parser_no_ocr.parse_pages(str(DATA / "sample_scanned_minutes.pdf"))
    print(f"Without OCR: pages with extractable text = {len(pages_no_ocr)}  (all blank, skipped)")

    # Now with OCR
    print("\nWith OCR fallback enabled:")
    try:
        parser_ocr = PDFParser(ocr_fallback=True)
        pages_ocr = parser_ocr.parse_pages(str(DATA / "sample_scanned_minutes.pdf"))
        print(f"Pages recovered: {len(pages_ocr)}")
        _hr()
        for p in pages_ocr:
            print(
                f"  page {p['page_num']}  "
                f"ocr_confidence={p.get('ocr_confidence')}%  "
                f"words={len(p['text'].split())}"
            )
            print(f"  preview: {p['text'][:140].strip()!r}")
            print()

        # Now ingest through full pipeline
        print("Ingesting through full pipeline (OCR mode):")
        vs = MemoryVectorSource()
        pipeline_ocr = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            ocr_fallback=True,
        )
        pipeline_ocr.ensure_index()
        records = pipeline_ocr.ingest(str(DATA / "sample_scanned_minutes.pdf"))
        print(f"  Chunks produced: {len(records)}")
        for r in records:
            print(
                f"  chunk {r['chunk_id'][-10:]}  "
                f"page={r.get('page_start', '-')}  "
                f"ocr={r['metadata'].get('ocr', False)}  "
                f"confidence={r['metadata'].get('ocr_confidence', '-')}"
            )

    except ImportError as e:
        print(f"  OCR deps not installed: {e}")
        print("  Run: pip install blue-data-doc-source[ocr]")
        print("       brew install tesseract poppler")


def step7_summary(pipeline):
    _section("Step 7 — Summary: what pypdf extracts vs what it silently skips")
    print(
        f"{'Format':<30} {'Text layer':<14} {'Images':<14} {'Table structure'}"
    )
    _hr()
    rows = [
        ("Invoice with table",      "yes (flat)",  "n/a",     "LOST — flat text"),
        ("Mixed text+image+table",  "yes (flat)",  "SKIPPED", "LOST — flat text"),
        ("Multi-page report",       "yes",         "SKIPPED", "LOST — flat text"),
        ("ArXiv research paper",    "yes",         "SKIPPED", "n/a"),
        ("Berkshire annual report", "yes",         "SKIPPED", "LOST — flat text"),
        ("Scanned minutes (OCR)",   "via OCR",     "n/a",     "n/a"),
    ]
    for name, text, imgs, tables in rows:
        print(f"  {name:<28} {text:<14} {imgs:<14} {tables}")

    print("""
Key takeaways:
  - pypdf extracts the text layer only; embedded images produce zero text
  - Table cells are extracted as flat prose with no row/column delimiters
  - Scanned (image-only) PDFs require OCR; set ocr_fallback=True on DocumentPipeline
  - OCR pages carry ocr=True and ocr_confidence in their metadata
  - Page provenance (page_start/page_end) is preserved on every chunk for all modes
""")


def run():
    _check_data_files()

    vs = MemoryVectorSource()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vs,
    )
    pipeline.ensure_index()

    step1_table_invoice(pipeline)
    step2_mixed_content(pipeline)
    step3_multipage(pipeline)
    step4_real_research(pipeline)
    step5_real_annual_report(pipeline)
    step6_scanned_ocr()
    step7_summary(pipeline)


if __name__ == "__main__":
    run()
