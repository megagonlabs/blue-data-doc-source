"""Retrieval quality benchmark: BM25 vs vector vs hybrid search modes.

Ingests a fixed 3-document corpus (no external files or API keys needed by
default) and evaluates retrieval quality on 24 hand-crafted ground-truth
query/chunk pairs.

Metrics:
  Hit@k  — fraction of queries where the relevant chunk appears in top-k results
  MRR    — mean reciprocal rank  (1/rank; 0 when not found in top-k_max)
  Latency — median query time (ms) per mode

Query types in the ground-truth set:
  keyword   — query tokens appear verbatim in the relevant chunk; BM25 should win
  paraphrase — shared key terms but rephrased; hybrid should win over pure BM25

Usage:
    python lib/tests/benchmarks/bench_retrieval.py
    python lib/tests/benchmarks/bench_retrieval.py --k 1 3 5 --runs 5
    python lib/tests/benchmarks/bench_retrieval.py --embedding openai   # OPENAI_API_KEY required
    python lib/tests/benchmarks/bench_retrieval.py --out RETRIEVAL_RESULTS.md
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))

from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel
from blue_data_doc_source.source.document.pipeline import DocumentPipeline

# ── Corpus ─────────────────────────────────────────────────────────────────────
# Three domain documents with clear section structure (ALL-CAPS headings).
# SectionChunker at chunk_size=100 produces one chunk per section body.

CORPUS: Dict[str, str] = {
    "insurance_policy.txt": """\
HEALTH INSURANCE POLICY

ELIGIBILITY
Coverage is available to full-time employees working at least thirty hours per week.
Dependents may be added during open enrollment or within thirty days of a qualifying life event.
Students enrolled full-time remain eligible as dependents until age twenty-six.

COVERAGE BENEFITS
Preventive care visits are fully covered with no copay required.
Emergency room visits require a one hundred fifty dollar copay per visit.
Prescription drugs are covered at eighty percent after the annual deductible is met.
Mental health services are covered at the same rate as physical health services.

DEDUCTIBLES AND OUT-OF-POCKET
The annual deductible is fifteen hundred dollars for individual coverage.
Family coverage has an annual deductible of three thousand dollars.
The maximum out-of-pocket limit is six thousand dollars per individual per year.

EXCLUSIONS
Cosmetic procedures are excluded unless deemed medically necessary by a physician.
Experimental treatments not approved by the FDA are excluded from coverage.
Dental and vision care require separate policy enrollment.
""",

    "service_agreement.txt": """\
SERVICE AGREEMENT

PAYMENT TERMS
The client shall pay a monthly retainer of five thousand dollars.
Invoices are due within thirty days of receipt.
Late payments incur a two percent penalty per month on the outstanding balance.
Wire transfers and ACH payments are the only accepted payment methods.

TERMINATION
Either party may terminate this agreement with sixty days written notice.
Immediate termination is permitted in cases of material breach or insolvency.
Upon termination all confidential materials must be returned within fourteen days.
Work completed prior to termination shall be compensated at the agreed rate.

INTELLECTUAL PROPERTY
All work product created under this agreement is owned by the client.
The service provider retains the right to use work as a portfolio example.
Pre-existing intellectual property of either party remains with its original owner.
Derivative works based on client data are exclusively owned by the client.

GOVERNING LAW
This agreement is governed by the laws of the state of Delaware.
Disputes shall be resolved through binding arbitration in Wilmington.
The prevailing party in any dispute is entitled to recover reasonable attorney fees.
""",

    "employee_handbook.txt": """\
EMPLOYEE HANDBOOK

WORK HOURS AND ATTENDANCE
Standard work hours are nine to five Monday through Friday.
Flexible schedules may be arranged with manager approval.
Employees must notify their manager at least one hour before a missed shift.
Remote work is permitted up to three days per week for eligible positions.

COMPENSATION AND BENEFITS
Salaries are reviewed annually during the performance review cycle in December.
Performance bonuses are paid quarterly based on individual and company metrics.
The company matches retirement contributions up to six percent of base salary.
Paid time off accrues at one and a half days per month for the first three years.

CODE OF CONDUCT
All employees must complete the annual ethics and compliance training.
Conflicts of interest must be disclosed to the human resources department.
Harassment or discrimination of any kind will not be tolerated and may result in termination.
All company data and systems must be used for legitimate business purposes only.

PERFORMANCE REVIEWS
Performance reviews are conducted twice yearly in June and December.
Employees receive written feedback within ten business days of each review.
A performance improvement plan is issued when performance falls below expectations.
Sustained underperformance after a plan may lead to termination of employment.
""",
}

# ── Ground truth ──────────────────────────────────────────────────────────────
# Each entry maps a query to a unique phrase that appears in the relevant chunk.
# find_relevant_chunk() resolves the phrase to a chunk_id after ingest.
#
# query_type:
#   "keyword"    — query tokens appear verbatim in chunk text / section title;
#                  BM25 should rank the relevant chunk #1
#   "paraphrase" — query shares key terms but is rephrased;
#                  hybrid should rank it higher than pure BM25

GROUND_TRUTH: List[Dict[str, Any]] = [
    # ── insurance_policy.txt ──────────────────────────────────────────────────
    {
        "query": "employee eligibility minimum hours per week",
        "relevant_phrase": "full-time employees working at least thirty hours",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "ELIGIBILITY",
    },
    {
        "query": "dependent coverage age limit students",
        "relevant_phrase": "eligible as dependents until age twenty-six",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "ELIGIBILITY",
    },
    {
        "query": "emergency room visit copay cost",
        "relevant_phrase": "one hundred fifty dollar copay per visit",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "COVERAGE BENEFITS",
    },
    {
        "query": "prescription drugs covered eighty percent deductible",
        "relevant_phrase": "covered at eighty percent after the annual deductible",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "COVERAGE BENEFITS",
    },
    {
        "query": "individual annual deductible amount fifteen hundred",
        "relevant_phrase": "annual deductible is fifteen hundred dollars for individual",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "DEDUCTIBLES AND OUT-OF-POCKET",
    },
    {
        "query": "maximum out-of-pocket limit six thousand dollars per year",
        "relevant_phrase": "maximum out-of-pocket limit is six thousand dollars",
        "query_type": "keyword",
        "doc": "insurance_policy.txt",
        "section": "DEDUCTIBLES AND OUT-OF-POCKET",
    },
    {
        "query": "what services are not covered by the health plan",
        "relevant_phrase": "Cosmetic procedures are excluded unless deemed medically",
        "query_type": "paraphrase",
        "doc": "insurance_policy.txt",
        "section": "EXCLUSIONS",
    },
    {
        "query": "free preventive checkups no out-of-pocket charge",
        "relevant_phrase": "Preventive care visits are fully covered with no copay",
        "query_type": "paraphrase",
        "doc": "insurance_policy.txt",
        "section": "COVERAGE BENEFITS",
    },
    # ── service_agreement.txt ─────────────────────────────────────────────────
    {
        "query": "monthly retainer fee five thousand dollars",
        "relevant_phrase": "monthly retainer of five thousand dollars",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "PAYMENT TERMS",
    },
    {
        "query": "invoice due date thirty days receipt",
        "relevant_phrase": "Invoices are due within thirty days of receipt",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "PAYMENT TERMS",
    },
    {
        "query": "late payment penalty two percent per month",
        "relevant_phrase": "two percent penalty per month on the outstanding balance",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "PAYMENT TERMS",
    },
    {
        "query": "termination notice period sixty days",
        "relevant_phrase": "terminate this agreement with sixty days written notice",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "TERMINATION",
    },
    {
        "query": "work product intellectual property owned by client",
        "relevant_phrase": "work product created under this agreement is owned by the client",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "INTELLECTUAL PROPERTY",
    },
    {
        "query": "governing law Delaware state agreement",
        "relevant_phrase": "governed by the laws of the state of Delaware",
        "query_type": "keyword",
        "doc": "service_agreement.txt",
        "section": "GOVERNING LAW",
    },
    {
        "query": "immediate contract cancellation breach insolvency",
        "relevant_phrase": "Immediate termination is permitted in cases of material breach",
        "query_type": "paraphrase",
        "doc": "service_agreement.txt",
        "section": "TERMINATION",
    },
    {
        "query": "how disputes are resolved arbitration",
        "relevant_phrase": "resolved through binding arbitration in Wilmington",
        "query_type": "paraphrase",
        "doc": "service_agreement.txt",
        "section": "GOVERNING LAW",
    },
    # ── employee_handbook.txt ─────────────────────────────────────────────────
    {
        "query": "remote work days permitted per week",
        "relevant_phrase": "Remote work is permitted up to three days per week",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "WORK HOURS AND ATTENDANCE",
    },
    {
        "query": "retirement contribution company match six percent salary",
        "relevant_phrase": "matches retirement contributions up to six percent of base salary",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "COMPENSATION AND BENEFITS",
    },
    {
        "query": "paid time off accrual rate per month",
        "relevant_phrase": "Paid time off accrues at one and a half days per month",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "COMPENSATION AND BENEFITS",
    },
    {
        "query": "performance reviews conducted June December twice yearly",
        "relevant_phrase": "reviews are conducted twice yearly in June and December",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "PERFORMANCE REVIEWS",
    },
    {
        "query": "harassment discrimination consequences termination",
        "relevant_phrase": "Harassment or discrimination of any kind will not be tolerated",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "CODE OF CONDUCT",
    },
    {
        "query": "salary compensation annual review cycle",
        "relevant_phrase": "Salaries are reviewed annually during the performance review cycle",
        "query_type": "keyword",
        "doc": "employee_handbook.txt",
        "section": "COMPENSATION AND BENEFITS",
    },
    {
        "query": "what happens after a failed performance improvement plan",
        "relevant_phrase": "Sustained underperformance after a plan may lead to termination",
        "query_type": "paraphrase",
        "doc": "employee_handbook.txt",
        "section": "PERFORMANCE REVIEWS",
    },
    {
        "query": "working from home schedule flexibility",
        "relevant_phrase": "Flexible schedules may be arranged with manager approval",
        "query_type": "paraphrase",
        "doc": "employee_handbook.txt",
        "section": "WORK HOURS AND ATTENDANCE",
    },
]

MODES = ["bm25", "vector", "hybrid"]
COLLECTION = "docs"
CHUNK_SIZE = 100
OVERLAP = 20


# ── Pipeline setup ────────────────────────────────────────────────────────────

def build_pipeline(embedding: str) -> DocumentPipeline:
    if embedding == "openai":
        from blue_data_doc_source.source.document.embeddings import OpenAIEmbeddingModel
        model = OpenAIEmbeddingModel()
    else:
        model = DummyEmbeddingModel(dim=8)
    vs = MemoryVectorSource()
    return DocumentPipeline(
        embedding_model=model,
        vector_source=vs,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        collection=COLLECTION,
    )


def ingest_corpus(pipeline: DocumentPipeline) -> int:
    pipeline.ensure_index(COLLECTION)
    total = 0
    for file_name, text in CORPUS.items():
        records = pipeline.ingest_text(
            text, file_name=file_name, mime_type="text/plain", collection=COLLECTION
        )
        total += len(records)
    return total


# ── Ground-truth resolution ───────────────────────────────────────────────────

def find_relevant_chunk(
    pipeline: DocumentPipeline, phrase: str
) -> Optional[str]:
    """Return the chunk_id whose text or section_title contains phrase (case-insensitive)."""
    needle = phrase.lower()
    for chunk_id, chunk in pipeline._chunk_index.items():
        text = (chunk.get("text") or "").lower()
        section = (chunk.get("section_title") or "").lower()
        if needle in text or needle in section:
            return chunk_id
    return None


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_mode(
    pipeline: DocumentPipeline,
    gt: List[Dict[str, Any]],
    mode: str,
    k_values: List[int],
    n_runs: int,
) -> Dict[str, Any]:
    max_k = max(k_values)
    hits: Dict[int, int] = {k: 0 for k in k_values}
    rr_scores: List[float] = []
    latencies: List[float] = []
    per_query: List[Dict[str, Any]] = []
    skipped = 0

    for entry in gt:
        relevant_id = find_relevant_chunk(pipeline, entry["relevant_phrase"])
        if relevant_id is None:
            skipped += 1
            continue

        times = []
        results: List[Dict] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            results = pipeline.search(
                entry["query"], k=max_k, collection=COLLECTION, mode=mode
            )
            times.append(time.perf_counter() - t0)
        latency_ms = statistics.median(times) * 1000
        latencies.append(latency_ms)

        result_ids = [r.get("chunk_id", "") for r in results]
        rank: Optional[int] = None
        for i, rid in enumerate(result_ids):
            if rid == relevant_id:
                rank = i + 1
                break

        for k in k_values:
            if rank is not None and rank <= k:
                hits[k] += 1
        rr_scores.append(1.0 / rank if rank is not None else 0.0)

        per_query.append({
            "query": entry["query"],
            "query_type": entry["query_type"],
            "doc": entry["doc"],
            "section": entry["section"],
            "rank": rank,
            "latency_ms": latency_ms,
        })

    n = len(per_query)
    return {
        "mode": mode,
        "n_queries": n,
        "skipped": skipped,
        "hit_at_k": {k: hits[k] / n if n else 0.0 for k in k_values},
        "mrr": statistics.mean(rr_scores) if rr_scores else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "per_query": per_query,
    }


def per_type_metrics(
    result: Dict[str, Any], k_values: List[int]
) -> Dict[str, Dict[str, Any]]:
    """Break down Hit@k and MRR by query_type."""
    by_type: Dict[str, List] = {}
    for q in result["per_query"]:
        qt = q["query_type"]
        by_type.setdefault(qt, []).append(q)

    out = {}
    for qt, queries in by_type.items():
        n = len(queries)
        rr = [1.0 / q["rank"] if q["rank"] else 0.0 for q in queries]
        hit_k = {}
        for k in k_values:
            hit_k[k] = sum(1 for q in queries if q["rank"] and q["rank"] <= k) / n
        out[qt] = {"n": n, "mrr": statistics.mean(rr), "hit_at_k": hit_k}
    return out


# ── Formatting ────────────────────────────────────────────────────────────────

W = 80


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


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def print_summary_table(
    results: List[Dict[str, Any]], k_values: List[int]
) -> None:
    k_headers = "  ".join(f"Hit@{k:1d}" for k in k_values)
    fmt = "  {:<10}  {:>5}  " + "  ".join(["{:>5}"] * len(k_values)) + "  {:>8}"
    print(fmt.format("Mode", "MRR", *[f"Hit@{k}" for k in k_values], "Lat(ms)"))
    print("  " + "─" * (10 + 5 + 8 * len(k_values) + 12))
    for r in results:
        hits = [_pct(r["hit_at_k"][k]) for k in k_values]
        print(fmt.format(
            r["mode"],
            f"{r['mrr']:.2f}",
            *hits,
            f"{r['median_latency_ms']:.1f}",
        ))


def print_per_type_table(
    results: List[Dict[str, Any]], k_values: List[int]
) -> None:
    query_types = sorted(
        {q["query_type"] for r in results for q in r["per_query"]}
    )
    for qt in query_types:
        print(f"\n  Query type: {qt}")
        fmt = "    {:<10}  {:>5}  " + "  ".join(["{:>5}"] * len(k_values))
        print(fmt.format("Mode", "MRR", *[f"Hit@{k}" for k in k_values]))
        print("    " + "─" * (10 + 5 + 7 * len(k_values) + 6))
        for r in results:
            by_type = per_type_metrics(r, k_values)
            m = by_type.get(qt)
            if m is None:
                continue
            hits = [_pct(m["hit_at_k"][k]) for k in k_values]
            print(fmt.format(r["mode"], f"{m['mrr']:.2f}", *hits))


def print_per_query_detail(results: List[Dict[str, Any]]) -> None:
    mode_rank: Dict[str, Dict[str, Optional[int]]] = {}
    for r in results:
        for q in r["per_query"]:
            key = q["query"]
            mode_rank.setdefault(key, {})
            mode_rank[key][r["mode"]] = q["rank"]

    # group by doc
    by_doc: Dict[str, List] = {}
    all_queries = results[0]["per_query"] if results else []
    for q in all_queries:
        by_doc.setdefault(q["doc"], []).append(q)

    for doc, queries in by_doc.items():
        print(f"\n  {doc}")
        hdr = f"    {'Query':<48}  {'Type':<10}  " + "  ".join(
            f"{m:>6}" for m in MODES
        )
        print(hdr)
        print("    " + "─" * (48 + 10 + 8 * len(MODES) + 8))
        for q in queries:
            key = q["query"]
            ranks = [mode_rank[key].get(m) for m in MODES]
            rank_str = [f"#{r}" if r else "  —" for r in ranks]
            query_short = (q["query"][:46] + "..") if len(q["query"]) > 48 else q["query"]
            row = f"    {query_short:<48}  {q['query_type']:<10}  " + "  ".join(
                f"{s:>6}" for s in rank_str
            )
            print(row)


# ── Markdown output ───────────────────────────────────────────────────────────

def _md_table_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def write_markdown(
    results: List[Dict[str, Any]],
    k_values: List[int],
    embedding: str,
    chunk_count: int,
    out_path: Path,
) -> None:
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Retrieval Quality Benchmark",
        f"",
        f"Date: {now}  |  Embedding: `{embedding}`  |  Corpus: {len(CORPUS)} docs, "
        f"{chunk_count} chunks  |  Ground truth: {len(GROUND_TRUTH)} queries",
        f"",
        f"## Overall results",
        f"",
    ]

    # header
    k_cols = [f"Hit@{k}" for k in k_values]
    header = _md_table_row("Mode", "MRR", *k_cols, "Latency (ms)")
    sep = _md_table_row(*["---"] * (3 + len(k_values)))
    lines += [header, sep]
    for r in results:
        hits = [_pct(r["hit_at_k"][k]) for k in k_values]
        lines.append(_md_table_row(
            f"`{r['mode']}`",
            f"{r['mrr']:.3f}",
            *hits,
            f"{r['median_latency_ms']:.1f}",
        ))

    lines += ["", "## By query type", ""]
    query_types = sorted({q["query_type"] for r in results for q in r["per_query"]})
    for qt in query_types:
        lines += [f"### {qt}", ""]
        header = _md_table_row("Mode", "N", "MRR", *k_cols)
        sep = _md_table_row(*["---"] * (3 + len(k_values)))
        lines += [header, sep]
        for r in results:
            by_type = per_type_metrics(r, k_values)
            m = by_type.get(qt)
            if m is None:
                continue
            hits = [_pct(m["hit_at_k"][k]) for k in k_values]
            lines.append(_md_table_row(
                f"`{r['mode']}`", str(m["n"]), f"{m['mrr']:.3f}", *hits
            ))
        lines.append("")

    lines += ["## Per-query detail", ""]
    for r in results:
        if r["mode"] != "hybrid":
            continue
        # just list queries with rank per mode
    mode_rank: Dict[str, Dict[str, Optional[int]]] = {}
    for r in results:
        for q in r["per_query"]:
            key = q["query"]
            mode_rank.setdefault(key, {})
            mode_rank[key][r["mode"]] = q["rank"]

    by_doc: Dict[str, List] = {}
    all_queries = results[0]["per_query"] if results else []
    for q in all_queries:
        by_doc.setdefault(q["doc"], []).append(q)

    for doc, queries in by_doc.items():
        lines += [f"### {doc}", ""]
        header = _md_table_row("Query", "Type", "Section", *[f"Rank ({m})" for m in MODES])
        sep = _md_table_row(*["---"] * (3 + len(MODES)))
        lines += [header, sep]
        for q in queries:
            key = q["query"]
            ranks = [mode_rank[key].get(m) for m in MODES]
            rank_str = [f"#{r}" if r else "—" for r in ranks]
            lines.append(_md_table_row(q["query"], q["query_type"], q["section"], *rank_str))
        lines.append("")

    lines += [
        "## Notes",
        "",
        f"- **BM25**: in-process keyword scoring with punctuation normalization and section-title boosting.",
        f"- **Vector**: KNN over MemoryVectorSource. With `DummyEmbeddingModel` vector similarity is "
        f"random — run with `--embedding openai` for meaningful vector/hybrid results.",
        f"- **Hybrid**: Reciprocal Rank Fusion (k=60) of BM25 and vector rankings.",
        f"- chunk_size={CHUNK_SIZE}, overlap={OVERLAP}; one section body per chunk.",
        f"- Hit@k = fraction of queries where relevant chunk appears in top-k.",
        f"- MRR = mean reciprocal rank; 1.0 = always rank 1, 0.0 = never found.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Results written to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval quality benchmark: bm25 vs vector vs hybrid")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                    help="k values for Hit@k (default: 1 3 5)")
    ap.add_argument("--runs", type=int, default=3,
                    help="Timed repetitions per query (median reported)")
    ap.add_argument("--embedding", choices=["dummy", "openai"], default="dummy",
                    help="Embedding model (dummy=no API key; openai=OPENAI_API_KEY required)")
    ap.add_argument("--out", type=Path, default=_HERE / "RETRIEVAL_RESULTS.md",
                    help="Path to write markdown results")
    args = ap.parse_args()

    k_values = sorted(set(args.k))
    embedding_label = "DummyEmbeddingModel(dim=8)" if args.embedding == "dummy" else "OpenAIEmbeddingModel"

    _banner(
        f"RETRIEVAL QUALITY BENCHMARK  |  embedding={args.embedding}  "
        f"k={k_values}  runs={args.runs}"
    )

    print(f"\n  Corpus: {len(CORPUS)} documents")
    print(f"  Ground truth: {len(GROUND_TRUTH)} queries  "
          f"({sum(1 for q in GROUND_TRUTH if q['query_type'] == 'keyword')} keyword, "
          f"{sum(1 for q in GROUND_TRUTH if q['query_type'] == 'paraphrase')} paraphrase)")
    print(f"  Embedding: {embedding_label}")
    if args.embedding == "dummy":
        print("  ⚠  DummyEmbeddingModel: vector similarity is random. "
              "Run --embedding openai for semantic quality.")

    print(f"\n  Building pipeline and ingesting corpus...", end="", flush=True)
    pipeline = build_pipeline(args.embedding)
    chunk_count = ingest_corpus(pipeline)
    print(f" done. ({chunk_count} chunks)")

    # Verify ground truth resolution
    missing = []
    for entry in GROUND_TRUTH:
        if find_relevant_chunk(pipeline, entry["relevant_phrase"]) is None:
            missing.append(entry["relevant_phrase"][:60])
    if missing:
        print(f"\n  WARNING: {len(missing)} ground-truth phrase(s) not found in any chunk:")
        for m in missing:
            print(f"    - {m!r}")

    _section("RESULTS BY MODE")
    print(f"  Searching {len(GROUND_TRUTH)} queries × {len(MODES)} modes "
          f"× {args.runs} runs each...\n")

    results = []
    for mode in MODES:
        print(f"  Running {mode}...", end="", flush=True)
        r = evaluate_mode(pipeline, GROUND_TRUTH, mode, k_values, args.runs)
        results.append(r)
        print(f" done  (MRR={r['mrr']:.2f}, "
              f"Hit@{max(k_values)}={_pct(r['hit_at_k'][max(k_values)])},"
              f" {r['median_latency_ms']:.1f}ms)")

    print()
    print_summary_table(results, k_values)

    _section("BY QUERY TYPE")
    print_per_type_table(results, k_values)

    _section("PER-QUERY DETAIL")
    print_per_query_detail(results)

    if args.out:
        write_markdown(results, k_values, args.embedding, chunk_count, args.out)

    # Key findings
    _section("KEY FINDINGS")
    bm25_r = next(r for r in results if r["mode"] == "bm25")
    vec_r = next(r for r in results if r["mode"] == "vector")
    hyb_r = next(r for r in results if r["mode"] == "hybrid")
    max_k = max(k_values)

    print(f"  BM25 Hit@{max_k}:   {_pct(bm25_r['hit_at_k'][max_k])}  MRR={bm25_r['mrr']:.2f}  "
          f"({bm25_r['median_latency_ms']:.1f}ms)")
    print(f"  Vector Hit@{max_k}: {_pct(vec_r['hit_at_k'][max_k])}  MRR={vec_r['mrr']:.2f}  "
          f"({vec_r['median_latency_ms']:.1f}ms)")
    print(f"  Hybrid Hit@{max_k}: {_pct(hyb_r['hit_at_k'][max_k])}  MRR={hyb_r['mrr']:.2f}  "
          f"({hyb_r['median_latency_ms']:.1f}ms)")

    if args.embedding == "dummy":
        print()
        print("  With DummyEmbeddingModel, vector results are effectively random.")
        print("  BM25 performance represents the keyword-only quality ceiling.")
        print("  Hybrid degrades BM25 quality because RRF incorporates random vector ranks.")
        print("  Run with --embedding openai to see hybrid outperform both BM25 and vector.")

    print()
    _rule("═")
    print("  Benchmark complete.")
    _rule("═")
    print()


if __name__ == "__main__":
    main()
