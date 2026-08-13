"""Chunker benchmark: SectionChunker vs WindowChunker vs SentenceChunker.

Measures throughput and chunk-quality statistics across three corpus types.
Can be run standalone or imported by runner.py.

Usage:
    python lib/tests/benchmarks/bench_chunkers.py
    python lib/tests/benchmarks/bench_chunkers.py --size medium --runs 10
"""
import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

from blue_data_doc_source.source.document.chunkers import (
    Chunker, SectionChunker, SentenceChunker, WindowChunker,
)

sys.path.insert(0, os.path.dirname(__file__))
from datasets import Dataset, build_corpus, SMALL, MEDIUM, LARGE


@dataclass
class ChunkerResult:
    chunker_name: str
    doc_type: str
    input_words: int
    chunk_count: int
    avg_words: float
    std_words: float
    min_words: int
    max_words: int
    titled_pct: float       # % of chunks that carry a section_title
    words_per_sec: float    # throughput: input words / median elapsed


def _time_chunker(
    chunker: Chunker, text: str, n_runs: int
) -> Tuple[List[Tuple], float]:
    """Return (chunks, median_elapsed_sec) over n_runs warm-up + timed runs."""
    # one untimed warm-up pass to avoid first-call penalty (e.g. regex compilation)
    result = chunker.chunk_with_titles(text)
    elapsed = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = chunker.chunk_with_titles(text)
        elapsed.append(time.perf_counter() - t0)
    return result, statistics.median(elapsed)


def benchmark_chunkers(
    text: str,
    doc_type: str,
    chunk_size: int = 200,
    overlap: int = 40,
    n_runs: int = 5,
) -> List[ChunkerResult]:
    """Benchmark all three chunkers on the same text. Returns one result per chunker."""
    input_words = len(text.split())
    chunkers = [
        ("SectionChunker",  SectionChunker(chunk_size=chunk_size, overlap=overlap)),
        ("WindowChunker",   WindowChunker(chunk_size=chunk_size, overlap=overlap)),
        ("SentenceChunker", SentenceChunker(chunk_size=chunk_size, overlap=overlap)),
    ]
    results = []
    for name, chunker in chunkers:
        chunks, elapsed = _time_chunker(chunker, text, n_runs)
        sizes = [len(ct.split()) for _, ct in chunks]
        titled = sum(1 for t, _ in chunks if t is not None)
        results.append(ChunkerResult(
            chunker_name=name,
            doc_type=doc_type,
            input_words=input_words,
            chunk_count=len(chunks),
            avg_words=statistics.mean(sizes) if sizes else 0.0,
            std_words=statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
            min_words=min(sizes) if sizes else 0,
            max_words=max(sizes) if sizes else 0,
            titled_pct=100.0 * titled / len(chunks) if chunks else 0.0,
            words_per_sec=input_words / elapsed if elapsed > 0 else 0.0,
        ))
    return results


# ── Standalone entry point ───────────────────────────────────────────────────

def _print_table(results: List[ChunkerResult], doc_type: str) -> None:
    rows = [r for r in results if r.doc_type == doc_type]
    if not rows:
        return
    r0 = rows[0]
    print(f"\nCorpus: {doc_type.upper()} ({r0.input_words:,} words)")
    fmt = "{:<20} {:>6}  {:>8}  {:>8}  {:>5}  {:>5}  {:>9}  {:>10}"
    print(fmt.format("Chunker", "Chunks", "Avg(wds)", "Std(wds)", "Min", "Max", "Section%", "Kwps"))
    print("─" * 76)
    for r in rows:
        print(fmt.format(
            r.chunker_name, r.chunk_count,
            f"{r.avg_words:.1f}", f"{r.std_words:.1f}",
            r.min_words, r.max_words,
            f"{r.titled_pct:.0f}%",
            f"{r.words_per_sec/1000:,.1f}K",
        ))
    print("─" * 76)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()

    target = {"small": SMALL, "medium": MEDIUM, "large": LARGE}[args.size]
    print(f"Building corpus ({args.size}, ~{target:,} words)...", flush=True)
    corpus = build_corpus(target)

    all_results: List[ChunkerResult] = []
    for ds in corpus:
        r = benchmark_chunkers(ds.text, ds.doc_type, args.chunk_size, args.overlap, args.runs)
        all_results.extend(r)
        _print_table(r, ds.doc_type)

    from collections import defaultdict
    by_chunker = defaultdict(list)
    for r in all_results:
        by_chunker[r.chunker_name].append(r.words_per_sec)
    avg = {n: sum(v) / len(v) for n, v in by_chunker.items()}
    winner = max(avg, key=avg.get)
    print(f"\nFastest overall: {winner}  ({avg[winner]/1000:,.1f}K wps avg)")
