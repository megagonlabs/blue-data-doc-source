"""Demo: pluggable parser layer — bring your own document processor.

Shows two approaches:

1. Parser ABC + LlamaIndex (--mode llamaindex)
   Implements the Parser ABC wrapping LlamaIndex's PDFReader. LlamaIndex
   handles extraction; our pipeline handles chunking, enrichment, and storage.
   Requires: pip install llama-index-core llama-index-readers-file

2. Bypass methods — ingest_text / ingest_pages (--mode bypass)
   Feeds pre-extracted content directly into the pipeline without going
   through any parser at all. No extra dependencies required.

Usage:
    # LlamaIndex parser (install llama-index first)
    pip install llama-index-core llama-index-readers-file
    python examples/demo_custom_parser.py --mode llamaindex

    # Bypass with pre-extracted content (no extra deps)
    python examples/demo_custom_parser.py --mode bypass

    # Use your own PDF
    python examples/demo_custom_parser.py --mode llamaindex path/to/doc.pdf
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from blue_doc_source import DocumentPipeline, Parser

DEFAULT_PDF = Path(__file__).parent.parent / "data" / "sample_policy.pdf"


# --- Approach 1: Parser ABC wrapping LlamaIndex ---

class LlamaIndexPDFParser(Parser):
    """Wraps LlamaIndex's PDFReader to conform to the Parser interface.

    LlamaIndex splits a PDF into one Document per page, each with full text
    and metadata. We map those directly to the page dicts the pipeline expects.
    """

    def parse(self, path: str) -> List[Dict[str, Any]]:
        try:
            from llama_index.readers.file import PDFReader
        except ImportError:
            print(
                "LlamaIndex not installed.\n"
                "Run: pip install llama-index-core llama-index-readers-file"
            )
            sys.exit(1)

        reader = PDFReader()
        docs = reader.load_data(file=Path(path))

        pages = []
        for doc in docs:
            page_num = doc.metadata.get("page_label")
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                page_num = None

            if doc.text.strip():
                pages.append({"page_num": page_num, "text": doc.text.strip()})

        return pages


# --- Approach 2: bypass — ingest_pages with pre-extracted content ---

def run_bypass(pipeline: DocumentPipeline, path: str) -> List[Dict[str, Any]]:
    """Simulate an external processor that already extracted page text."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_num": i, "text": text})

    print(f"Pre-extracted {len(pages)} pages — feeding directly via ingest_pages()")
    return pipeline.ingest_pages(pages, file_name=Path(path).name)


# --- Main ---

def print_results(records: List[Dict[str, Any]], top_k: int = 5) -> None:
    print(f"\n{len(records)} chunks produced. First {min(top_k, len(records))}:\n")
    for r in records[:top_k]:
        page = f"p{r['page_start']}" if r.get("page_start") else ""
        section = f"  [{r['section_title']}]" if r.get("section_title") else ""
        snippet = r["text"][:120].replace("\n", " ")
        print(f"  {r['chunk_id']}  {page}{section}")
        print(f"  {snippet}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom parser demo")
    parser.add_argument(
        "path", nargs="?", default=str(DEFAULT_PDF), help="Path to a .pdf file"
    )
    parser.add_argument(
        "--mode",
        choices=["llamaindex", "bypass"],
        default="bypass",
        help="llamaindex: use LlamaIndex PDFReader via Parser ABC; "
             "bypass: feed pre-extracted pages via ingest_pages()",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    print(f"File   : {args.path}")
    print(f"Mode   : {args.mode}")
    print(f"Chunks : size={args.chunk_size}, overlap={args.overlap}\n")

    if args.mode == "llamaindex":
        print("Using LlamaIndexPDFParser (Parser ABC) ...")
        pipeline = DocumentPipeline(
            parser=LlamaIndexPDFParser(),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        records = pipeline.ingest(args.path)
    else:
        print("Using ingest_pages() bypass ...")
        pipeline = DocumentPipeline(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        records = run_bypass(pipeline, args.path)

    print_results(records)
