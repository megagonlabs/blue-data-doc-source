"""Parsers Demo
=============
Shows the raw output of every built-in parser before it enters the pipeline.
Most demos show what chunks look like *after* ingestion. This one shows what
each parser.parse() / parse_blocks() actually returns.

Four parsers covered:
  TextParser       -- .txt  -- one string, no pages
  PDFParser        -- .pdf  -- list of page dicts; OCR fallback for scanned pages
  DocxParser       -- .docx -- one string, headings -> markdown #, tables -> pipes
  PlumberPDFParser -- .pdf  -- typed blocks: text blocks + structured table blocks

Run:
    python3 examples/feature-demos/parsers_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data_doc_source import PlumberPDFParser
from blue_data_doc_source.source.document.parsers.pdf_parser import PDFParser
from blue_data_doc_source.source.document.parsers.text_parser import TextParser
from blue_data_doc_source.source.document.parsers.docx_parser import DocxParser

DATA = Path(__file__).parent.parent.parent / "data"

DIV  = "=" * 72
DIV2 = "-" * 72


def _banner(title: str) -> None:
    print()
    print(DIV)
    print(f"  {title}")
    print(DIV)


def _step(n: int, title: str) -> None:
    print()
    print(DIV2)
    print(f"  Step {n} -- {title}")
    print(DIV2)


def _preview(text: str, chars: int = 200) -> str:
    text = text.replace("\n", " ↵ ")
    return text[:chars] + ("..." if len(text) > chars else "")


def run() -> None:
    _banner("Parsers Demo — raw parser output before the pipeline")

    # ── Step 1: TextParser ────────────────────────────────────────────────
    _step(1, "TextParser  (.txt)  —  one string, no pages")

    path = DATA / "sample_policy.txt"
    text = TextParser().parse(str(path))

    word_count = len(text.split())
    line_count = text.count("\n")

    print()
    print("  File:", path.name)
    print(f"  Returns: str  ({word_count} words, {line_count} lines)")
    print()
    print("  TextParser.parse(path) ->")
    print(f'    "{_preview(text, 300)}"')
    print()
    print("  Key behaviour:")
    print("    - Reads the entire file as UTF-8")
    print("    - Returns a plain string — no page structure, no metadata")
    print("    - Pipeline wraps it as [{\"text\": full_content}]")
    print("    - page_start / page_end will be None on every chunk")

    # ── Step 2: PDFParser — digital PDF ──────────────────────────────────
    _step(2, "PDFParser  (.pdf)  —  list of page dicts, flat text")

    path = DATA / "sample_contract.pdf"
    pages = PDFParser().parse_pages(str(path))

    print()
    print("  File:", path.name)
    print(f"  Returns: list[dict]  ({len(pages)} pages)")
    print()
    print("  PDFParser().parse_pages(path) ->")
    for p in pages:
        words = len(p["text"].split())
        preview = _preview(p["text"], 120)
        print(f"    [{{'page_num': {p['page_num']}, 'text': '{preview}'")
        print(f"       words={words}}}]")
    print()
    print("  Key behaviour:")
    print("    - Uses pypdf; one dict per page with 'page_num' (1-based)")
    print("    - Blank pages (no text layer) are silently skipped")
    print("    - Tables extracted as flat prose — row/column structure is LOST")
    print("    - Images embedded in the PDF produce zero text (silently skipped)")

    # ── Step 3: PDFParser — invoice with table ────────────────────────────
    _step(3, "PDFParser  (.pdf)  —  table lost as flat text")

    path = DATA / "sample_table_invoice.pdf"
    pages = PDFParser().parse_pages(str(path))

    print()
    print("  File:", path.name, "(contains a structured table of line items)")
    print(f"  Returns: list[dict]  ({len(pages)} pages)")
    print()
    print("  PDFParser().parse_pages(path) ->")
    for p in pages:
        preview = _preview(p["text"], 250)
        print(f"    page {p['page_num']}: '{preview}'")
    print()
    print("  The table rows appear as flat prose — no | delimiters, no structure.")
    print("  Use PlumberPDFParser (Step 5) to preserve table structure.")

    # ── Step 4: PDFParser — scanned PDF with OCR ─────────────────────────
    _step(4, "PDFParser  (.pdf, scanned)  —  OCR fallback")

    path = DATA / "sample_scanned_minutes.pdf"

    # Without OCR
    pages_no_ocr = PDFParser(ocr_fallback=False).parse_pages(str(path))
    print()
    print("  File:", path.name, "(image-only PDF — no text layer)")
    print()
    print("  Without OCR (default):")
    print(f"    PDFParser().parse_pages(path) -> {pages_no_ocr!r}")
    print("    Empty list — all pages blank, all skipped.")
    print()

    # With OCR
    try:
        pages_ocr = PDFParser(ocr_fallback=True).parse_pages(str(path))
        print("  With OCR fallback:")
        print(f"    PDFParser(ocr_fallback=True).parse_pages(path) -> {len(pages_ocr)} page(s)")
        print()
        for p in pages_ocr:
            words = len(p["text"].split())
            preview = _preview(p["text"], 150)
            print(f"    [{{")
            print(f"      'page_num':       {p['page_num']},")
            print(f"      'text':           '{preview}',")
            print(f"      'ocr':            {p.get('ocr')},")
            print(f"      'ocr_confidence': {p.get('ocr_confidence')}   # mean Tesseract word confidence 0-100")
            print(f"    }}]")
        print()
        print("  Key behaviour:")
        print("    - Page images preprocessed: grayscale -> contrast 2x -> binarise")
        print("    - ocr=True marks Tesseract-extracted pages in chunk metadata")
        print("    - ocr_confidence propagates to ChunkRecord.metadata['ocr_confidence']")
    except ImportError:
        print("  With OCR fallback: (skipped — install OCR deps to see this)")
        print("    pip install blue-data-doc-source[ocr]")
        print("    brew install tesseract poppler")

    # ── Step 5: DocxParser ────────────────────────────────────────────────
    _step(5, "DocxParser  (.docx)  —  headings to markdown, tables to pipes")

    path = DATA / "sample_policy.docx"
    text = DocxParser().parse(str(path))

    lines = text.split("\n\n")
    print()
    print("  File:", path.name)
    print(f"  Returns: str  ({len(text.split())} words, {len(lines)} blocks)")
    print()
    print("  DocxParser().parse(path) -> (first 12 blocks shown)")
    print()
    for i, block in enumerate(lines[:12]):
        display = block.replace("\n", " ↵ ")
        print(f"  [{i:02d}] {display[:100]}")
    print()
    print("  Heading conversion:")
    headings = [ln for ln in text.split("\n") if ln.startswith("#")]
    for h in headings:
        print(f"    {h}")
    print()
    print("  Table rendering (first table found):")
    table_lines = [ln for ln in text.split("\n") if ln.startswith("|")]
    for tl in table_lines[:6]:
        print(f"    {tl}")
    if len(table_lines) > 6:
        print(f"    ... ({len(table_lines) - 6} more rows)")
    print()
    print("  Key behaviour:")
    print("    - Iterates doc.element.body XML to preserve paragraph+table order")
    print("    - Heading 1 -> #, Heading 2 -> ##, Heading 3 -> ###, etc.")
    print("    - Tables rendered as | cell | cell | rows (not structured dicts)")
    print("    - Returns one string — no page_num — pipeline treats as single page")
    print("    - SectionChunker detects the # headers and assigns section_title")

    # ── Step 6: PlumberPDFParser — parse_blocks() ─────────────────────────
    _step(6, "PlumberPDFParser  (.pdf)  —  typed blocks: text + structured tables")

    path = DATA / "sample_table_invoice.pdf"

    try:
        blocks = PlumberPDFParser().parse_blocks(str(path))

        text_blocks  = [b for b in blocks if b["type"] == "text"]
        table_blocks = [b for b in blocks if b["type"] == "table"]

        print()
        print("  File:", path.name, "(same invoice used in Step 3)")
        print(f"  Returns: list[dict]  ({len(blocks)} blocks: {len(text_blocks)} text, {len(table_blocks)} table)")
        print()
        print("  PlumberPDFParser().parse_blocks(path) ->")

        for b in blocks:
            print()
            if b["type"] == "text":
                preview = _preview(b["text"], 120)
                print(f"  {{")
                print(f"    'type':     'text',")
                print(f"    'page_num': {b.get('page_num')},")
                print(f"    'text':     '{preview}'")
                print(f"  }}")
            else:
                tbl = b["table"]
                print(f"  {{")
                print(f"    'type':     'table',")
                print(f"    'page_num': {b.get('page_num')},")
                print(f"    'text':     '{_preview(b['text'], 120)}',")
                print(f"    'table': {{")
                print(f"      'headers': {tbl['headers']},")
                print(f"      'rows': [")
                for row in tbl["rows"]:
                    print(f"        {row},")
                print(f"      ]")
                print(f"    }}")
                print(f"  }}")

        print()
        print("  Contrast with Step 3 (PDFParser): the table is now structured.")
        print("  In the pipeline:")
        print("    - text blocks  -> chunker -> one or more ChunkRecord(content_type='text')")
        print("    - table blocks -> bypass chunker -> one ChunkRecord(content_type='table')")
        print("    - content_type is indexed as TAG -> filterable in search")

    except ImportError:
        print()
        print("  (skipped — pdfplumber not installed)")
        print("  pip install pdfplumber")

    # ── Step 7: PlumberPDFParser — parse() backward compat ────────────────
    _step(7, "PlumberPDFParser  —  parse() flattens blocks to page dicts")

    try:
        pages = PlumberPDFParser().parse(str(path))
        print()
        print("  PlumberPDFParser().parse(path) -> (backward-compat; merges all blocks per page)")
        print()
        for p in pages:
            words = len(p["text"].split())
            preview = _preview(p["text"], 180)
            print(f"    [{{")
            print(f"      'page_num': {p.get('page_num')},")
            print(f"      'text':     '{preview}',")
            print(f"      words={words}")
            print(f"    }}]")
        print()
        print("  Same interface as PDFParser.parse_pages() — drop-in replacement")
        print("  when you don't need block-level content_type on chunks.")

    except ImportError:
        pass

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary — parser return types")
    print()
    print("  Parser              Input    Returns            Page-aware  Tables")
    print("  " + "-" * 68)
    print("  TextParser          .txt     str                no          flat text")
    print("  PDFParser           .pdf     list[page dict]    yes         flat text")
    print("  PDFParser + OCR     .pdf     list[page dict]    yes         flat text")
    print("  DocxParser          .docx    str                no          pipe rows")
    print("  PlumberPDFParser    .pdf     list[block dict]   yes         structured")
    print()
    print("  All parsers feed into DocumentPipeline.ingest() transparently.")
    print("  The chunker, enricher, and embedder see the same interface regardless")
    print("  of which parser was used.")
    print()
    print("  PlumberPDFParser is the only parser that returns typed blocks.")
    print("  Use it when you need:")
    print("    - table.headers / table.rows for downstream processing")
    print("    - content_type filtering in vector search")
    print("    - one-chunk-per-table guarantee (tables bypass the chunker)")


if __name__ == "__main__":
    run()
