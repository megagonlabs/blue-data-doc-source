"""Master benchmark runner — builds the corpus once, runs all benchmarks, prints tables.

Usage:
    python lib/tests/benchmarks/runner.py
    python lib/tests/benchmarks/runner.py --size small --runs 3
    python lib/tests/benchmarks/runner.py --size large --runs 10 --chunk-size 300 --overlap 60
    python lib/tests/benchmarks/runner.py --skip-retrieval   # skip retrieval quality benchmark
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from datasets import build_corpus, SMALL, MEDIUM, LARGE
from bench_chunkers import benchmark_chunkers, ChunkerResult
from bench_pipeline import benchmark_chunk_size_sweep, benchmark_pipeline_stages, SweepResult, StageResult
from bench_retrieval import (
    build_pipeline, ingest_corpus, evaluate_mode, print_summary_table,
    print_per_type_table, write_markdown, GROUND_TRUTH, MODES, _HERE,
)

W = 80


# ── Formatting ───────────────────────────────────────────────────────────────

def _rule(char="─", w=W):
    print(char * w)


def _banner(title: str):
    print()
    _rule("═")
    print(f"  {title}")
    _rule("═")


def _section(label: str):
    print()
    tail = "─" * max(0, W - len(label) - 4)
    print(f"── {label} {tail}")


def _kwps(wps: float) -> str:
    return f"{wps / 1_000:,.1f}K"


# ── Table: chunker comparison ────────────────────────────────────────────────

def _print_chunker_table(results: List[ChunkerResult], doc_type: str) -> None:
    rows = [r for r in results if r.doc_type == doc_type]
    if not rows:
        return
    print(f"\n  Corpus: {doc_type.upper()} ({rows[0].input_words:,} words)")
    fmt = "  {:<20} {:>6}  {:>8}  {:>8}  {:>5}  {:>5}  {:>9}  {:>9}"
    print(fmt.format("Chunker", "Chunks", "Avg(wds)", "Std(wds)", "Min", "Max", "Section%", "Kwps"))
    print("  " + "─" * 74)
    for r in rows:
        print(fmt.format(
            r.chunker_name,
            r.chunk_count,
            f"{r.avg_words:.1f}",
            f"{r.std_words:.1f}",
            r.min_words,
            r.max_words,
            f"{r.titled_pct:.0f}%",
            _kwps(r.words_per_sec),
        ))
    print("  " + "─" * 74)


# ── Table: chunk-size sweep ──────────────────────────────────────────────────

def _print_sweep_table(results: List[SweepResult]) -> None:
    fmt = "  {:>10}  {:>7}  {:>6}  {:>8}  {:>8}  {:>15}"
    print(fmt.format("chunk_size", "overlap", "Chunks", "Avg(wds)", "Std(wds)", "Throughput(Kwps)"))
    print("  " + "─" * 62)
    for r in results:
        print(fmt.format(
            r.chunk_size, r.overlap,
            r.chunk_count,
            f"{r.avg_words:.1f}",
            f"{r.std_words:.1f}",
            _kwps(r.words_per_sec),
        ))


# ── Table: pipeline stage breakdown ─────────────────────────────────────────

def _print_stage_table(s: StageResult) -> None:
    fmt = "  {:<25}  {:>10}  {:>9}  {:>12}"
    print(fmt.format("Stage", "Time(ms)", "% total", "µs/chunk"))
    print("  " + "─" * 62)
    rows = [
        ("Chunking",          s.chunk_ms),
        ("Enrichment",        s.enrich_ms),
        ("Embedding (dummy)", s.embed_ms),
    ]
    for name, ms in rows:
        pct = f"{100 * ms / s.total_ms:.1f}%" if s.total_ms > 0 else "n/a"
        per = f"{1_000 * ms / s.chunk_count:.1f}" if s.chunk_count else "n/a"
        print(fmt.format(name, f"{ms:.2f}", pct, per))
    print("  " + "─" * 62)
    per_total = f"{1_000 * s.total_ms / s.chunk_count:.1f}" if s.chunk_count else "n/a"
    print(fmt.format("Total", f"{s.total_ms:.2f}", "100.0%", per_total))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run all blue-data-doc-source benchmarks")
    ap.add_argument("--size",       choices=["small", "medium", "large"], default="medium",
                    help="Corpus size (small≈2K, medium≈10K, large≈50K words)")
    ap.add_argument("--runs",       type=int, default=5,
                    help="Timed repetitions per measurement (median is reported)")
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="chunk_size used for chunker comparison and stage breakdown")
    ap.add_argument("--overlap",    type=int, default=40,
                    help="overlap used for chunker comparison and stage breakdown")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="Skip the retrieval quality benchmark (benchmarks 1-3 only)")
    ap.add_argument("--embedding",  choices=["dummy", "openai"], default="dummy",
                    help="Embedding model for retrieval benchmark (openai requires OPENAI_API_KEY)")
    args = ap.parse_args()

    target      = {"small": SMALL, "medium": MEDIUM, "large": LARGE}[args.size]
    chunk_size  = args.chunk_size
    overlap     = args.overlap
    n_runs      = args.runs

    _banner(
        f"blue-data-doc-source BENCHMARK SUITE  |  "
        f"size={args.size}  chunk_size={chunk_size}  overlap={overlap}  runs={n_runs}"
    )

    print(f"\n  Building {args.size} corpus (~{target:,} words × 3 doc types)...", end="", flush=True)
    corpus = build_corpus(target)
    print(" done.\n")
    for ds in corpus:
        print(f"    {ds.doc_type:<14} {ds.word_count:>7,} words")

    # ── 1. Chunker comparison ────────────────────────────────────────────────
    _section("1. CHUNKER COMPARISON")
    print("  SectionChunker / WindowChunker / SentenceChunker on three corpus types.")
    print("  Section%: chunks that carry a detected section_title.")
    print("  Kwps: thousands of input words processed per second (median of runs).")

    all_chunker: List[ChunkerResult] = []
    for ds in corpus:
        results = benchmark_chunkers(
            text=ds.text,
            doc_type=ds.doc_type,
            chunk_size=chunk_size,
            overlap=overlap,
            n_runs=n_runs,
        )
        all_chunker.extend(results)
        _print_chunker_table(results, ds.doc_type)

    by_chunker = defaultdict(list)
    for r in all_chunker:
        by_chunker[r.chunker_name].append(r.words_per_sec)
    avg_wps = {n: sum(v) / len(v) for n, v in by_chunker.items()}
    winner = max(avg_wps, key=avg_wps.get)
    print(f"\n  Fastest across all corpus types: {winner} ({_kwps(avg_wps[winner])} wps avg)")

    # ── 2. Chunk-size sweep ──────────────────────────────────────────────────
    structured_ds = next(ds for ds in corpus if ds.doc_type == "structured")

    _section("2. CHUNK SIZE SWEEP  (SectionChunker, structured corpus)")
    print("  How chunk_size affects chunk count, average size, and throughput.")
    print()
    sweep = benchmark_chunk_size_sweep(structured_ds.text, n_runs=n_runs)
    _print_sweep_table(sweep)

    # ── 3. Pipeline stage breakdown ──────────────────────────────────────────
    _section(f"3. PIPELINE STAGE BREAKDOWN  (SectionChunker, structured, {structured_ds.word_count:,} words)")
    print(f"  chunk_size={chunk_size}  overlap={overlap}")
    print("  Embedding uses DummyEmbeddingModel (SHA-256, no network) to isolate library cost.")
    print()
    stages = benchmark_pipeline_stages(
        structured_ds.text,
        chunk_size=chunk_size,
        overlap=overlap,
        n_runs=n_runs,
    )
    _print_stage_table(stages)

    # ── 4. Retrieval quality ─────────────────────────────────────────────────
    if not args.skip_retrieval:
        _section("4. RETRIEVAL QUALITY  (BM25 vs vector vs hybrid)")
        print(f"  {len(GROUND_TRUTH)} ground-truth queries on a 3-document domain corpus.")
        print(f"  Embedding: {args.embedding}  (vector similarity is random with dummy)")
        print()
        ret_pipeline = build_pipeline(args.embedding)
        chunk_count = ingest_corpus(ret_pipeline)
        print(f"  Ingested {chunk_count} chunks. Running {len(MODES)} modes...\n")
        ret_results = []
        for mode in MODES:
            r = evaluate_mode(ret_pipeline, GROUND_TRUTH, mode, [1, 3, 5], n_runs)
            ret_results.append(r)
        print_summary_table(ret_results, [1, 3, 5])
        print_per_type_table(ret_results, [1, 3, 5])
        write_markdown(ret_results, [1, 3, 5], args.embedding, chunk_count,
                       _HERE / "RETRIEVAL_RESULTS.md")

    print()
    _rule("═")
    print("  Benchmark complete.")
    _rule("═")
    print()


if __name__ == "__main__":
    main()
