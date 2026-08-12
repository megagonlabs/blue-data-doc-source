# Benchmarks

The benchmark suite measures throughput and behaviour of the chunking pipeline on synthetic corpora. It covers three areas: chunker comparison, chunk-size sweep, and per-stage pipeline breakdown.

---

## Running the benchmarks

```bash
# Default: medium corpus (~10K words), 5 timed runs
python lib/tests/benchmarks/runner.py

# Larger corpus, more repetitions
python lib/tests/benchmarks/runner.py --size large --runs 10

# Custom chunk size
python lib/tests/benchmarks/runner.py --size medium --runs 7 --chunk-size 300 --overlap 60
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--size` | `medium` | Corpus size: `small` (~2K words), `medium` (~10K), `large` (~50K) |
| `--runs` | `5` | Timed repetitions per measurement. Median is reported. |
| `--chunk-size` | `200` | Chunk size used for the chunker comparison and stage breakdown |
| `--overlap` | `40` | Overlap used for the chunker comparison and stage breakdown |

The runner does one untimed warm-up pass before timed runs to avoid first-call overhead from regex compilation and import resolution.

---

## Corpus

The suite generates a deterministic synthetic corpus with three document types, each built to the target word count:

| Type | Structure | Description |
|---|---|---|
| `structured` | Heavy | Markdown headers at multiple levels, numbered sections, varied paragraph lengths |
| `unstructured` | None | Continuous prose paragraphs, no section markers |
| `mixed` | Light | Mix of headed sections and plain prose blocks |

The corpus is generated fresh each run from a fixed seed, so results are reproducible across machines. No external files or network access required.

---

## Benchmarks

### 1. Chunker comparison

Runs `SectionChunker`, `WindowChunker`, and `SentenceChunker` on each corpus type. Reports:

| Column | Description |
|---|---|
| Chunks | Total chunks produced |
| Avg(wds) | Mean words per chunk |
| Std(wds) | Standard deviation of words per chunk |
| Min / Max | Smallest and largest chunk in words |
| Section% | Percentage of chunks carrying a detected `section_title` |
| Kwps | Input words processed per second (thousands), median across runs |

### 2. Chunk-size sweep

Runs `SectionChunker` on the structured corpus across five `chunk_size` / `overlap` configurations (50→1000 words). Shows how chunk count, average size, and throughput change as the window grows.

### 3. Pipeline stage breakdown

Times each stage of the pipeline independently on the structured corpus:

- **Chunking** — `SectionChunker.chunk_with_titles()`
- **Enrichment** — `BasicEnricher.enrich()` on each chunk
- **Embedding** — `DummyEmbeddingModel.embed_text()` (SHA-256, no network) to isolate in-process library cost

Reports time in milliseconds, percentage of total, and microseconds per chunk.

---

## Baseline results

Reference run on a MacBook Pro M-series, Python 3.12, medium corpus, 7 timed repetitions.

```
python lib/tests/benchmarks/runner.py --size medium --runs 7
```

### Chunker comparison

**Structured corpus (10,462 words)**

```
Chunker              Chunks  Avg(wds)  Std(wds)    Min    Max   Section%       Kwps
SectionChunker           97     113.3      66.9      4    200       100%  14,986.8K
WindowChunker            66     197.9      17.0     62    200         0%  23,600.7K
SentenceChunker          66     193.5       6.6    151    200         0%   6,068.7K
```

**Unstructured corpus (10,090 words)**

```
Chunker              Chunks  Avg(wds)  Std(wds)    Min    Max   Section%       Kwps
SectionChunker           64     196.6      24.0     10    200         0%  18,519.4K
WindowChunker            64     196.6      24.0     10    200         0%  24,233.0K
SentenceChunker          64     192.8       6.0    151    200         0%   6,379.7K
```

**Mixed corpus (10,191 words)**

```
Chunker              Chunks  Avg(wds)  Std(wds)    Min    Max   Section%       Kwps
SectionChunker           80     144.4      63.2      3    200       100%  15,305.6K
WindowChunker            64     198.6      11.1    111    200         0%  25,222.7K
SentenceChunker          64     194.2       3.9    188    200         0%   6,050.6K
```

### Chunk-size sweep (SectionChunker, structured corpus)

```
chunk_size  overlap  Chunks  Avg(wds)  Std(wds)  Throughput(Kwps)
        50       10     282      43.1      13.5        12,338.5K
       100       20     159      74.2      32.6        14,315.2K
       200       40      97     113.3      66.9        14,656.9K
       500      100      64     157.3      46.1        15,499.3K
      1000      150      64     157.3      46.1        15,695.0K
```

### Pipeline stage breakdown (structured, 10,462 words, chunk_size=200)

```
Stage                        Time(ms)    % total      µs/chunk
Chunking                         0.68       5.4%           7.0
Enrichment                      11.81      93.7%         121.8
Embedding (dummy)                0.11       0.9%           1.1
Total                           12.61     100.0%         130.0
```

---

## Key findings

**Chunker speed:** `WindowChunker` is fastest at ~24M wps — pure word-split sliding window with no regex overhead. `SectionChunker` is ~40% slower on structured text due to per-line header detection; it matches `WindowChunker` speed on unstructured text (same code path when no headers found). `SentenceChunker` is 4–5× slower than `WindowChunker` — sentence tokenisation is the bottleneck regardless of whether nltk or regex is used.

**Section coverage:** `SectionChunker` is the only chunker that propagates `section_title`. On structured and mixed corpora it achieves 100% coverage. On unstructured text it falls back to window chunking and produces identical results to `WindowChunker` (0% section coverage is expected behaviour, not a failure).

**Chunk size consistency:** `WindowChunker` has near-zero `Std(wds)` — chunks are exactly `chunk_size` words except the final one. `SentenceChunker` has low std (~4–7 wds) because sentences are short and uniform. `SectionChunker` has high std (47–67 wds) because it respects section boundaries that vary in length; this is intentional.

**Dominant pipeline cost:** `BasicEnricher` consumes ~94% of total in-process pipeline time at ~122 µs/chunk. Chunking costs only 7 µs/chunk. If enrichment output is not used downstream, disabling `BasicEnricher` is a ~10× end-to-end speedup. Real OpenAI embeddings will cost ~50–150 ms per API call and will completely dominate the pipeline — in-process stages are negligible by comparison.

**Chunk-size sweep:** Smaller chunks (size=50) produce more chunks and lower average word counts but slightly lower throughput (~20% drop) because the `SectionChunker` header scan is called once per line regardless of chunk size. Above `chunk_size=500` on this corpus, the document is too small to split further so chunk count plateaus.

---

## Adding a new baseline

After a significant change — new chunker, enricher rewrite, Python version upgrade — capture a new baseline:

```bash
python lib/tests/benchmarks/runner.py --size medium --runs 7 > lib/tests/benchmarks/baseline.txt
```

Include the machine spec and Python version as a comment at the top of the file so results are comparable across environments.
