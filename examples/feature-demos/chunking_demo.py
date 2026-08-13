"""
Chunking Deep-Dive
==================
Shows how SectionChunker, WindowChunker, and SentenceChunker behave on the
same document — and where they differ.

Key facts demonstrated:
  • chunk_size and overlap are WORD COUNTS, not character counts
  • SectionChunker detects 4 header patterns; falls back to WindowChunker
  • WindowChunker stride = chunk_size − overlap; last chunk may be shorter
  • SentenceChunker splits on sentence boundaries first, then windows
  • Section titles propagate to every sub-chunk produced within that section

Run:
    python3 examples/feature-demos/chunking_demo.py
    python3 examples/feature-demos/chunking_demo.py --chunk-size 150 --overlap 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

from blue_data_doc_source import SectionChunker, SentenceChunker, WindowChunker

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
DIV  = "=" * 68
DIV2 = "-" * 68


def word_count(text: str) -> int:
    return len(text.split())


def show_chunks(
    chunks: list[tuple[str | None, str]],
    label: str,
    max_show: int = 6,
) -> None:
    print(f"\n  {label}  →  {len(chunks)} chunks")
    print(f"  {DIV2}")
    for i, (title, text) in enumerate(chunks[:max_show]):
        wc      = word_count(text)
        snippet = text[:90].replace("\n", " ")
        title_s = f"  [{title}]" if title else ""
        print(f"  chunk {i:02d}  words={wc:4d}{title_s}")
        print(f"           {snippet}…")
    if len(chunks) > max_show:
        print(f"  … {len(chunks) - max_show} more chunks")


def run(doc_path: str, chunk_size: int, overlap: int) -> None:
    text = Path(doc_path).read_text(encoding="utf-8")
    total_words = word_count(text)

    print()
    print(DIV)
    print("Chunking Deep-Dive")
    print(DIV)
    print()
    print(f"  Document   : {doc_path}")
    print(f"  Total words: {total_words}")
    print(f"  chunk_size : {chunk_size} words")
    print(f"  overlap    : {overlap} words")
    print(f"  stride     : {chunk_size - overlap} words  (WindowChunker)")
    print()

    # ── WindowChunker ─────────────────────────────────────────────────────
    print(DIV2)
    print("WindowChunker — sliding word window, no section awareness")
    print(DIV2)
    window = WindowChunker(chunk_size=chunk_size, overlap=overlap)
    w_chunks = window.chunk_with_titles(text)
    show_chunks(w_chunks, "WindowChunker")
    print()
    print("  All titles are None (WindowChunker never assigns titles).")
    print(f"  Expected chunk count ≈ {max(1, (total_words - overlap) // (chunk_size - overlap))}")

    # ── SectionChunker ────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("SectionChunker — splits on headers, windows within each section")
    print(DIV2)
    section = SectionChunker(chunk_size=chunk_size, overlap=overlap)
    s_chunks = section.chunk_with_titles(text)
    show_chunks(s_chunks, "SectionChunker")

    titled = [(t, tx) for t, tx in s_chunks if t is not None]
    print()
    print(f"  {len(titled)} of {len(s_chunks)} chunks carry a section title.")
    print()
    print("  Header patterns matched (first 5 unique titles):")
    seen: set[str] = set()
    for title, _ in s_chunks:
        if title and title not in seen:
            seen.add(title)
            print(f"    • {title}")
        if len(seen) >= 5:
            break

    # ── SentenceChunker ───────────────────────────────────────────────────
    print()
    print(DIV2)
    print("SentenceChunker — splits on sentence boundaries first, then windows")
    print(DIV2)
    sentence = SentenceChunker(chunk_size=chunk_size, overlap=overlap)
    sent_chunks = sentence.chunk_with_titles(text)
    show_chunks(sent_chunks, "SentenceChunker")

    # ── Word count proof ──────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Word-count proof  (chunk_size is words, not characters)")
    print(DIV2)
    print()
    small = WindowChunker(chunk_size=10, overlap=2)
    tiny_text = "One two three four five six seven eight nine ten eleven twelve."
    tiny_chunks = small.chunk_with_titles(tiny_text)
    for i, (_, tx) in enumerate(tiny_chunks):
        wc = word_count(tx)
        print(f"  chunk {i}: {wc:2d} words  →  \"{tx}\"")
    print()
    print(f"  chunk_size=10, overlap=2 → stride=8 → {len(tiny_chunks)} chunks")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(DIV)
    print("Summary")
    print(DIV)
    print()
    print(f"  {'Chunker':<20}  {'Chunks':>6}  {'Section-aware':>14}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*14}")
    print(f"  {'WindowChunker':<20}  {len(w_chunks):>6}  {'No':>14}")
    print(f"  {'SectionChunker':<20}  {len(s_chunks):>6}  {'Yes':>14}")
    print(f"  {'SentenceChunker':<20}  {len(sent_chunks):>6}  {'No':>14}")
    print()
    print("  SectionChunker is the default. It falls back to WindowChunker")
    print("  when no headers are detected in the document.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunking deep-dive")
    parser.add_argument("path",          nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--chunk-size",  type=int,  default=200)
    parser.add_argument("--overlap",     type=int,  default=30)
    args = parser.parse_args()
    run(args.path, args.chunk_size, args.overlap)
