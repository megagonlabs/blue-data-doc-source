"""
Demo 1 — Section-Aware Chunking vs Sliding Window
===================================================
Problem : WindowChunker cuts by word count alone.  It can split a subsection
          mid-thought, mixing content from two unrelated topics in one chunk.
          A query about "prescription drugs" might land in a chunk that's half
          "prescription drugs" and half "emergency room" — diluting the signal.

Solution: SectionChunker detects document structure (markdown headers, numbered
          sections, ALL-CAPS headings, underlined headings) and ensures each chunk
          stays within a single logical section.  Retrieval quality improves
          because chunks are topically coherent.

Result  : On sample_policy.txt — SectionChunker produces section-aligned chunks
          where every chunk carries a title.  WindowChunker produces raw word
          windows that frequently straddle section boundaries.

Run:
    python3 examples/scenario-demos/section_vs_window_demo.py
    python3 examples/scenario-demos/section_vs_window_demo.py --chunk-size 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

from blue_doc_source import SectionChunker, WindowChunker

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
DIV  = "=" * 68
DIV2 = "-" * 68

PROBE_QUERIES = [
    "prescription drug",
    "emergency room",
    "pre-existing condition",
    "deductible",
    "enrollment period",
]


def word_count(text: str) -> int:
    return len(text.split())


def topic_overlap(chunk_text: str, topics: list[str]) -> list[str]:
    """Return which topics appear in this chunk."""
    lower = chunk_text.lower()
    return [t for t in topics if t in lower]


def print_chunks(
    chunks: list[tuple[str | None, str]],
    label: str,
    topics: list[str],
    max_show: int = 8,
) -> None:
    print(f"\n  {label}  →  {len(chunks)} chunks")
    print(f"  {DIV2}")
    cross_boundary = 0
    for i, (title, text) in enumerate(chunks[:max_show]):
        matched = topic_overlap(text, topics)
        wc      = word_count(text)
        snippet = text[:100].replace("\n", " ")
        title_s = f"  [{title}]" if title else "  [no title]"
        topics_s = f"  topics={matched}" if matched else ""
        multi    = "  ⚠ multi-topic" if len(matched) > 1 else ""
        if len(matched) > 1:
            cross_boundary += 1
        print(f"  chunk {i:02d}  words={wc:4d}{title_s}{topics_s}{multi}")
        print(f"           {snippet}…")
    if len(chunks) > max_show:
        print(f"  … {len(chunks) - max_show} more chunks")
    print()
    print(f"  Chunks with multiple topics in one chunk: {cross_boundary} of {min(max_show, len(chunks))} shown")


def run(doc_path: str, chunk_size: int, overlap: int) -> None:
    text = Path(doc_path).read_text(encoding="utf-8")
    topics = PROBE_QUERIES

    print()
    print(DIV)
    print("Demo 1 — Section-Aware Chunking vs Sliding Window")
    print(DIV)
    print()
    print(f"  Document  : {doc_path}")
    print(f"  Words     : {word_count(text)}")
    print(f"  chunk_size: {chunk_size} words")
    print(f"  overlap   : {overlap} words")
    print()
    print("  Probe topics (to detect cross-section contamination):")
    for t in topics:
        print(f"    • {t}")
    print()

    # ── WindowChunker ─────────────────────────────────────────────────────
    print(DIV2)
    print("WindowChunker — raw word windows")
    print(DIV2)
    window   = WindowChunker(chunk_size=chunk_size, overlap=overlap)
    w_chunks = window.chunk_with_titles(text)
    print_chunks(w_chunks, "WindowChunker", topics)

    # ── SectionChunker ────────────────────────────────────────────────────
    print(DIV2)
    print("SectionChunker — section-aligned windows")
    print(DIV2)
    section  = SectionChunker(chunk_size=chunk_size, overlap=overlap)
    s_chunks = section.chunk_with_titles(text)
    print_chunks(s_chunks, "SectionChunker", topics)

    # ── Section title inventory ────────────────────────────────────────────
    print(DIV2)
    print("Section titles detected by SectionChunker")
    print(DIV2)
    print()
    seen: dict[str, int] = {}
    for title, _ in s_chunks:
        if title:
            seen[title] = seen.get(title, 0) + 1
    for title, count in seen.items():
        print(f"  {title:<45}  {count} chunk{'s' if count > 1 else ''}")

    # ── Retrieval relevance comparison ────────────────────────────────────
    print()
    print(DIV2)
    print("Retrieval relevance  —  which chunk contains 'prescription drug'?")
    print(DIV2)
    print()

    query = "prescription drug"

    w_hits = [(i, t, tx) for i, (t, tx) in enumerate(w_chunks) if query in tx.lower()]
    s_hits = [(i, t, tx) for i, (t, tx) in enumerate(s_chunks) if query in tx.lower()]

    print(f"  WindowChunker  : {len(w_hits)} chunk(s) contain '{query}'")
    for i, title, text_chunk in w_hits[:3]:
        other = [t for t in topics if t in text_chunk.lower() and t != query]
        print(f"    chunk {i:02d}  [no title]  also contains: {other}")
        print(f"      {text_chunk[:120].replace(chr(10), ' ')}…")
    print()

    print(f"  SectionChunker : {len(s_hits)} chunk(s) contain '{query}'")
    for i, title, text_chunk in s_hits[:3]:
        other = [t for t in topics if t in text_chunk.lower() and t != query]
        print(f"    chunk {i:02d}  [{title}]  also contains: {other}")
        print(f"      {text_chunk[:120].replace(chr(10), ' ')}…")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(DIV)
    print("Result")
    print(DIV)
    print()

    w_cross = sum(1 for _, tx in w_chunks if len(topic_overlap(tx, topics)) > 1)
    s_cross = sum(1 for _, tx in s_chunks if len(topic_overlap(tx, topics)) > 1)

    print(f"  {'Chunker':<20}  {'Chunks':>6}  {'Multi-topic chunks':>18}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*18}")
    print(f"  {'WindowChunker':<20}  {len(w_chunks):>6}  {w_cross:>18}")
    print(f"  {'SectionChunker':<20}  {len(s_chunks):>6}  {s_cross:>18}")
    print()
    print("  SectionChunker chunks are topically coherent because each chunk")
    print("  stays within one section.  Retrieval finds the right section")
    print("  without noise from adjacent topics.")
    print()
    print("  WindowChunker is faster to compute and is the right choice for")
    print("  unstructured text that has no discernible headers.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section-aware vs window chunking")
    parser.add_argument("path",          nargs="?", default=str(DEFAULT_DOC))
    parser.add_argument("--chunk-size",  type=int,  default=200)
    parser.add_argument("--overlap",     type=int,  default=30)
    args = parser.parse_args()
    run(args.path, args.chunk_size, args.overlap)
