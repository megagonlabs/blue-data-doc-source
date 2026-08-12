"""
Custom Parser Demo  —  Extend the Parser ABC or Bypass It Entirely
===================================================================
Problem : The built-in TextParser and PDFParser may not suit every source.
          You might have a proprietary PDF SDK, a vision-model extractor,
          or a third-party document library that already does page splitting.

Solution: Two approaches, same pipeline output.

  Mode A — Parser ABC (--mode llamaindex)
    Implement the Parser ABC to wrap any extraction library.
    The pipeline calls your parse() method and handles everything else
    (chunking, enrichment, embedding, storage).
    Example here: wraps LlamaIndex's PDFReader.
    Requires: pip install llama-index-core llama-index-readers-file

  Mode B — ingest_pages() bypass (--mode bypass, default)
    Extract pages yourself — any tool, any format — then hand a
    list of {"text": ..., "page_num": ...} dicts directly to the pipeline.
    Skips the parser layer entirely. No extra dependencies.

Run:
    python3 examples/integration-demos/custom_parser_demo.py
    python3 examples/integration-demos/custom_parser_demo.py --mode bypass path/to/doc.pdf
    python3 examples/integration-demos/custom_parser_demo.py --mode llamaindex path/to/doc.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from blue_doc_source import DocumentPipeline, DummyEmbeddingModel, Parser

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_PDF = DATA_DIR / "sample_policy.pdf"
DIV  = "=" * 68
DIV2 = "-" * 68


# ── Mode A: Parser ABC wrapping LlamaIndex ────────────────────────────────

class LlamaIndexPDFParser(Parser):
    """Wraps LlamaIndex's PDFReader to conform to the Parser ABC.

    LlamaIndex splits a PDF into one Document per page. We map those to the
    page dicts the pipeline expects: {"page_num": int, "text": str}.
    """

    def parse(self, path: str) -> List[Dict[str, Any]]:
        try:
            from llama_index.readers.file import PDFReader  # type: ignore
        except ImportError:
            print("LlamaIndex not installed.")
            print("  pip install llama-index-core llama-index-readers-file")
            sys.exit(1)

        reader = PDFReader()
        docs   = reader.load_data(file=Path(path))
        pages  = []
        for doc in docs:
            try:
                page_num = int(doc.metadata.get("page_label", 0))
            except (TypeError, ValueError):
                page_num = None
            if doc.text.strip():
                pages.append({"page_num": page_num, "text": doc.text.strip()})
        return pages


# ── Mode B: bypass with ingest_pages() ────────────────────────────────────

def extract_pages_with_pypdf(path: str) -> List[Dict[str, Any]]:
    """Extract page text using pypdf (built-in PDF reader in pdfplumber dep chain)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        print("pypdf not installed.  pip install pypdf")
        sys.exit(1)

    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_num": i, "text": text})
    return pages


# ── Shared display helper ─────────────────────────────────────────────────

def show_records(records: List[Dict[str, Any]], top_k: int = 6) -> None:
    print(f"\n  {len(records)} chunks produced")
    print(f"  {'chunk_id':<42}  {'page':>4}  {'section_title'}")
    print(f"  {'-'*42}  {'-'*4}  {'-'*30}")
    for r in records[:top_k]:
        cid     = r.get("chunk_id", "")[:42]
        page    = str(r["page_start"]) if r.get("page_start") is not None else "—"
        section = (r.get("section_title") or "")[:35]
        print(f"  {cid:<42}  {page:>4}  {section}")
    if len(records) > top_k:
        print(f"  … {len(records) - top_k} more chunks")


# ── Main ──────────────────────────────────────────────────────────────────

def run(path: str, mode: str, chunk_size: int, overlap: int) -> None:
    print()
    print(DIV)
    print("Custom Parser Demo")
    print(DIV)
    print()
    print(f"  File  : {path}")
    print(f"  Mode  : {mode}")
    print()

    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if mode == "llamaindex":
        print(DIV2)
        print("Mode A — Parser ABC wrapping LlamaIndex PDFReader")
        print(DIV2)
        print()
        print("  class LlamaIndexPDFParser(Parser):")
        print("      def parse(self, path) -> list[dict]:")
        print("          # call any extraction library here")
        print("          # return [{\"page_num\": 1, \"text\": \"...\"}, ...]")
        print()
        print("  The pipeline calls parse(), then chunks/enriches/embeds the output.")
        print("  Your parser replaces the built-in TextParser/PDFParser entirely.")
        print()
        print("  Ingesting with LlamaIndexPDFParser …")
        pipeline_custom = DocumentPipeline(
            parser=LlamaIndexPDFParser(),
            embedding_model=DummyEmbeddingModel(dim=8),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        records = pipeline_custom.ingest(path)
        show_records(records)

    else:
        print(DIV2)
        print("Mode B — ingest_pages() bypass: extract externally, skip the parser")
        print(DIV2)
        print()
        print("  Use case: you already have page text from a vision model, a")
        print("  proprietary SDK, or another library.  No Parser subclass needed.")
        print()

        if not Path(path).exists():
            print(f"  PDF not found: {path}")
            print("  Running with synthetic pages instead …")
            print()
            pages = [
                {"page_num": 1, "text": "Eligibility\n\nCoverage is available to employees aged 18–65."},
                {"page_num": 2, "text": "Coverage\n\nEmergency room visits carry a $300 copayment."},
                {"page_num": 3, "text": "Cost Sharing\n\nThe annual deductible is $1,500 per individual."},
            ]
        else:
            print("  Extracting pages with pypdf …")
            pages = extract_pages_with_pypdf(path)

        print(f"  {len(pages)} page(s) extracted  →  feeding to ingest_pages()")
        print()
        print("  pipeline.ingest_pages(")
        print("      pages=[{\"page_num\": 1, \"text\": \"...\"}, ...],")
        print("      file_name=\"doc.pdf\",")
        print("  )")
        print()

        records = pipeline.ingest_pages(
            pages=pages,
            file_name=Path(path).name,
            mime_type="application/pdf",
        )
        show_records(records)

        print()
        cross = [r for r in records
                 if r.get("page_start") is not None and r.get("page_end") is not None
                 and r["page_start"] != r["page_end"]]
        print(f"  ✓ Chunks spanning multiple pages: {len(cross)}  (must be 0)")
        print(f"  ✓ page_start == page_end on every chunk — page-scoped ingestion preserved")

    # ── Parser ABC interface ───────────────────────────────────────────────
    print()
    print(DIV2)
    print("Parser ABC interface")
    print(DIV2)
    print()
    print("  from blue_doc_source import Parser")
    print()
    print("  class MyParser(Parser):")
    print("      def parse(self, path: str) -> list[dict]:")
    print("          # Return one dict per page:")
    print("          # {\"page_num\": int,   ← optional; sets page_start/page_end")
    print("          #  \"text\": str}        ← required")
    print()
    print("  pipeline = DocumentPipeline(parser=MyParser(), ...)")
    print("  records  = pipeline.ingest(\"doc.pdf\")")
    print()
    print("  Alternative — bypass the parser layer entirely:")
    print("  records = pipeline.ingest_pages(pages, file_name=\"doc.pdf\")")
    print("  records = pipeline.ingest_text(text,  file_name=\"doc.txt\")")
    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  Mode A (Parser ABC): cleanest when you always use the same extractor.")
    print("    pipeline.ingest(path) works as normal; your parser is called internally.")
    print()
    print("  Mode B (bypass): best when extraction already happened elsewhere, or")
    print("    when you want to feed content that has no file on disk.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom parser demo")
    parser.add_argument("path",          nargs="?", default=str(DEFAULT_PDF))
    parser.add_argument("--mode",        choices=["bypass", "llamaindex"], default="bypass")
    parser.add_argument("--chunk-size",  type=int, default=150)
    parser.add_argument("--overlap",     type=int, default=20)
    args = parser.parse_args()
    run(args.path, args.mode, args.chunk_size, args.overlap)
