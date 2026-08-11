# Benchmarks

Performance benchmarks for the `blue-data-doc-source` pipeline. Three focused
benchmarks measure chunker throughput, chunk-size trade-offs, and per-stage
pipeline costs.

## Running

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

### Full suite (recommended starting point)

```bash
python lib/tests/benchmarks/runner.py                           # medium corpus, 5 runs
python lib/tests/benchmarks/runner.py --size small  --runs 3   # quick smoke-check
python lib/tests/benchmarks/runner.py --size large  --runs 10  # high-confidence numbers
python lib/tests/benchmarks/runner.py --chunk-size 300 --overlap 60   # custom config
```

### Individual benchmarks

```bash
python lib/tests/benchmarks/bench_chunkers.py --size medium --runs 7
python lib/tests/benchmarks/bench_pipeline.py --size medium --runs 7
```

### Options

| Flag           | Default  | Description                                          |
|----------------|----------|------------------------------------------------------|
| `--size`       | `medium` | Corpus size: `small` ≈2K, `medium` ≈10K, `large` ≈50K words |
| `--runs`       | `5`      | Timed repetitions per measurement (median reported)  |
| `--chunk-size` | `200`    | Words per chunk (chunker comparison + stage breakdown) |
| `--overlap`    | `40`     | Overlap words between consecutive chunks             |

---

## What each benchmark measures

### 1. Chunker comparison

Runs `SectionChunker`, `WindowChunker`, and `SentenceChunker` on three
synthetically generated corpus types — **structured** (markdown headers +
numbered subsections), **unstructured** (dense prose), and **mixed** (numbered
sections alternating with plain paragraphs).

Columns:

| Column     | Meaning                                                             |
|------------|---------------------------------------------------------------------|
| `Chunks`   | Number of chunks produced from the input text                       |
| `Avg(wds)` | Mean words per chunk                                                |
| `Std(wds)` | Standard deviation of chunk word count — lower = more uniform       |
| `Min/Max`  | Smallest and largest chunk in words                                 |
| `Section%` | Percentage of chunks that carry a `section_title` on `ChunkRecord` |
| `Kwps`     | Thousands of input words processed per second (median elapsed)      |

### 2. Chunk-size sweep

Runs `SectionChunker` at five `chunk_size` settings (`50 → 1000`) on the
structured corpus and reports how chunk count, average size, and throughput
scale. Use this to choose a `chunk_size` that balances retrieval granularity
against processing cost.

### 3. Pipeline stage breakdown

Times chunking, enrichment (`BasicEnricher`), and embedding
(`DummyEmbeddingModel`) independently on the same text. `DummyEmbeddingModel`
is used intentionally — OpenAI API latency would dominate and hide the
library's own cost. Replace it with `OpenAIEmbeddingModel` locally if you want
to measure real embedding throughput.

---

## Corpus types

The synthetic corpus is generated deterministically (fixed seed, no external
files required) by `datasets.py`. Three types exercise different code paths:

| Type           | Description                                            | Best chunker       |
|----------------|--------------------------------------------------------|--------------------|
| `structured`   | Markdown `# headers` and `## numbered subsections`    | `SectionChunker`   |
| `unstructured` | Dense prose, no headers                                | `WindowChunker` / `SentenceChunker` |
| `mixed`        | Numbered `1.` sections alternating with prose blocks   | `SectionChunker`   |

---

## Reference baseline

`baseline.txt` contains a full medium-corpus run captured on 2026-08-10
(MacBook Pro M-series, Python 3.12, 7 runs). Update it after any change that
intentionally affects performance:

```bash
python lib/tests/benchmarks/runner.py --size medium --runs 7 > lib/tests/benchmarks/baseline.txt
```

Then review the diff to confirm the change moved numbers in the expected direction.

---

## Key findings from the baseline run

**Chunker speed:** `WindowChunker` is the fastest (~24M wps) — it is a pure
word-split sliding window with no regex overhead. `SectionChunker` is ~40%
slower on structured text due to per-line header detection; it matches
`WindowChunker` speed on unstructured text (no headers found, same fallback
path). `SentenceChunker` is 4–5× slower than `WindowChunker` due to sentence
tokenisation overhead.

**Section coverage:** `SectionChunker` is the only chunker that propagates
`section_title` to `ChunkRecord`. On structured and mixed corpora it achieves
100% section coverage, which is essential for agent explainability.

**Stage breakdown — primary finding:** `BasicEnricher` consumes ~94% of total
pipeline time at ~122 µs/chunk. Chunking costs only 7 µs/chunk. If enrichment
output is not used downstream, disabling `BasicEnricher` is a ~10× end-to-end
speedup. Real OpenAI embeddings (~50–150 ms/call) will completely dominate the
pipeline when enabled.

**Chunk-size sweep:** Throughput drops ~20% at `chunk_size=50` vs `chunk_size=200`
because the `SectionChunker` header scan runs per-line regardless of chunk size.
Above `chunk_size=500` the corpus is too small to split further; chunk count
plateaus.
