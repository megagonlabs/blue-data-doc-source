# blue-data-doc-source — Testing Strategy

## Overview

`blue-data-doc-source` uses three test tiers for the core library and separate adapter tests for each platform service:

- **Unit tests** — run by default (`testpaths = ["lib/tests/unit"]`); all external I/O mocked or replaced with in-memory implementations
- **Integration tests** — require Redis Stack on `localhost:6379`; run explicitly
- **Benchmarks** — chunker throughput, pipeline throughput, retrieval quality; recorded results committed to `benchmarks/RETRIEVAL_RESULTS.md`

---

## 1. Tooling

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage measurement |

Both are declared as `dev` dependencies. No async test library is needed — the pipeline is synchronous.

---

## 2. Test Layout

```
lib/tests/
├── unit/                              # 20 test files; all I/O mocked or in-memory
│   ├── test_basic_enricher.py         # BasicEnricher: PROPER_NOUN regex, semantic_hints
│   ├── test_document_find.py          # DOCUMENT.FIND structured filter logic
│   ├── test_document_pipeline.py      # DocumentPipeline: ingest, search, delete, find, graph ops
│   ├── test_document_record.py        # ChunkRecord: to_dict() field omission, chunk_id format
│   ├── test_document_source.py        # DocumentDataSource: op dispatch, all 9 ops
│   ├── test_docx_parser.py            # DocxParser: heading preservation, paragraph extraction
│   ├── test_graph_indexer.py          # GraphIndexer: node/edge creation, dedup, delete
│   ├── test_hybrid_search.py          # Hybrid BM25 + vector search; pre-filter integration
│   ├── test_llm_enricher.py           # LLMEnricher: mocked openai client; entity + hints output
│   ├── test_nl_parser.py              # NLQueryParser: mocked Anthropic; op dict construction
│   ├── test_openai_embedding.py       # OpenAIEmbeddingModel: mocked openai; dim validation
│   ├── test_pdf_parser.py             # PDFParser: digital PDF; OCR fallback mocked
│   ├── test_plumber_pdf_parser.py     # PlumberPDFParser: text + table block output
│   ├── test_pre_filtering.py          # Redis Search filter expression builder
│   ├── test_rich_search.py            # DOCUMENT.SEARCH rich=True: provenance + neighbor context
│   ├── test_section_chunker.py        # SectionChunker: header detection, fallback, word-count sizing
│   ├── test_sentence_chunker.py       # SentenceChunker: sentence boundaries; mocked nltk
│   ├── test_structured_extractor.py   # StructuredFieldExtractor: Pydantic + TypedDict + JSON Schema
│   ├── test_text_parser.py            # TextParser: UTF-8 read, empty file
│   └── test_window_chunker.py         # WindowChunker: overlap, chunk_size, word-count correctness
├── integration/
│   ├── conftest.py                    # Redis connection fixture; skip when Redis unavailable
│   └── test_end_to_end.py             # full ingest → search → find → delete round-trip; RedisVectorSource
└── benchmarks/
    ├── README.md
    ├── RETRIEVAL_RESULTS.md           # recorded P@k / MRR baselines
    ├── bench_chunkers.py              # throughput: docs/sec through SectionChunker vs WindowChunker
    ├── bench_pipeline.py              # throughput: docs/sec through full DocumentPipeline
    ├── bench_retrieval.py             # retrieval quality: P@k, MRR on curated query/document pairs
    ├── datasets.py                    # fixture documents and ground-truth relevance sets
    └── runner.py                      # benchmark harness

platform/document-mcp/tests/
└── test_server.py                     # MCP tool handlers; patched DocumentPipeline singleton

platform/cli/tests/
└── test_cli.py                        # Click CliRunner; patched DocumentPipeline singleton
```

---

## 3. In-Process Test Doubles

### In-memory backends

Unit tests substitute the production backends with in-process alternatives:

| Production dependency | Test double | Why |
|----------------------|-------------|-----|
| `RedisVectorSource` | `MemoryVectorSource` | No Redis process required; same `VectorSource` interface |
| `RedisJSONSource` | `MemoryJSONSource` | No Redis process required; same `JSONSource` interface |
| `Neo4JGraphSource` | `MemoryGraphSource` | No Neo4j process required; same `GraphSource` interface |
| `FileRecordStore` | `MemoryRecordStore` | No filesystem I/O; same `RecordStore` interface |

These are real implementations of the same ABCs, not mocks. The same code paths run in tests and production — only the backend differs.

### Mocked LLM / embedding clients

`openai.OpenAI` and `anthropic.Anthropic` are mocked via `unittest.mock.patch` in unit tests. Tests assert on:
- The text sent to the API (prompt content, schema shape)
- How the response is parsed into entities, semantic hints, or extracted fields

```python
@patch("blue_data_doc_source.source.document.enrichment.llm_enricher.openai.OpenAI")
def test_llm_enricher_returns_entities(mock_client):
    mock_client.return_value.chat.completions.create.return_value = _mock_response(
        '{"entities": [{"text": "Acme Corp", "type": "ORG"}], "semantic_hints": {}}'
    )
    enricher = LLMEnricher()
    result = enricher.enrich("Acme Corp signed the agreement.")
    assert result["entities"][0]["text"] == "Acme Corp"
```

### DummyEmbeddingModel

`DummyEmbeddingModel` produces deterministic fixed-dimension vectors from a SHA-256 hash of the input text. It requires no API key, runs in microseconds, and is used in all unit tests and most integration tests. Embedding similarity scores are not meaningful with `DummyEmbeddingModel` — tests assert on structure, field presence, and call behaviour, not on ranking quality.

---

## 4. Unit Tests

Unit tests cover each module in isolation. Tests assert on:

- Return value structure and field presence
- `chunk_id` / `document_id` format and determinism
- `to_dict()` field omission: `None` optional fields must be absent, not null
- Parser output structure: list of page dicts with required keys
- Chunker word-count boundaries: chunks do not exceed `chunk_size` words
- Section detection: heading patterns trigger section splits; fallback fires when no headings present
- `DocumentDataSource.execute_query` op dispatch: all 9 ops route correctly
- Error conditions: unsupported ops, missing required fields, unparseable documents

**Example — ChunkRecord ID determinism:**

```python
def test_chunk_id_is_deterministic(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("Hello world.")
    pipeline = DocumentPipeline(embedding_model=DummyEmbeddingModel(), vector_source=MemoryVectorSource())
    chunks_a = pipeline.ingest(str(txt))
    chunks_b = pipeline.ingest(str(txt))
    assert chunks_a[0]["chunk_id"] == chunks_b[0]["chunk_id"]
```

**Example — to_dict() omits None fields:**

```python
def test_to_dict_omits_none_page_start():
    record = ChunkRecord(
        document_id="abc", chunk_id="abc_chunk_0000",
        text="hello", file_name="doc.txt",
    )
    d = record.to_dict()
    assert "page_start" not in d
    assert "page_end" not in d
    assert "section_title" not in d
```

---

## 5. Integration Tests

`lib/tests/integration/test_end_to_end.py` exercises the full stack against a real Redis Stack instance. It requires `docker run -p 6379:6379 redis/redis-stack-server:latest` (or equivalent) before running.

The integration suite verifies:
- `DOCUMENT.ENSURE_INDEX` creates the vector index with all extra TAG/NUMERIC fields
- `DOCUMENT.INGEST` upserts all chunks to Redis; re-ingest is idempotent
- `DOCUMENT.SEARCH` returns relevant chunks by KNN; pre-filters on `document_id` and `mime_type` narrow results
- `DOCUMENT.FIND` matches DocumentRecords by structured field filters; dot-notation and operator filters work
- `DOCUMENT.FIND_THEN_SEARCH` correctly chains the two operations
- `DOCUMENT.DELETE` removes all chunks for a document from the vector index

`conftest.py` skips the entire integration suite when Redis is unavailable — the test process exits cleanly without failures.

---

## 6. Benchmarks

Benchmarks are not part of the default test run and are not tracked by pytest. They are standalone scripts that record results to `benchmarks/RETRIEVAL_RESULTS.md`.

| Benchmark | What it measures |
|-----------|----------------|
| `bench_chunkers.py` | Throughput: documents/sec through SectionChunker vs WindowChunker on a fixed corpus |
| `bench_pipeline.py` | Throughput: documents/sec through the full `DocumentPipeline` (parse + chunk + enrich + embed) |
| `bench_retrieval.py` | Retrieval quality: P@1, P@3, P@5, MRR on curated (query, relevant_chunk_id) pairs |

The retrieval benchmark uses `DummyEmbeddingModel` to isolate chunking quality from embedding quality — P@k scores reflect how well the chunker preserves relevance-determining content, not the embedding model's semantic quality. A separate run with `OpenAIEmbeddingModel` is included in `RETRIEVAL_RESULTS.md` for comparison.

---

## 7. Platform Adapter Tests

### MCP (`platform/document-mcp/tests/test_server.py`)

The `DocumentPipeline` singleton is patched. Tests assert that each MCP tool call dispatches to the correct `pipeline.*` method with the expected arguments.

### CLI (`platform/cli/tests/test_cli.py`)

Uses Click's `CliRunner`. The `DocumentPipeline` singleton is patched. Tests assert on:
- Exit codes (0 for success; non-zero for missing required args or pipeline errors)
- `pipeline.*` method call arguments
- Stdout format (plain text summary and `--json` JSON output)

---

## 8. Running Tests

```bash
# Unit tests (default — no external services)
.venv/bin/pytest lib/tests/unit/ -v

# Single test file
.venv/bin/pytest lib/tests/unit/test_section_chunker.py -v

# Integration tests (requires Redis Stack on localhost:6379)
docker run -p 6379:6379 redis/redis-stack-server:latest
.venv/bin/pytest lib/tests/integration/ -v

# CLI adapter tests
.venv/bin/pytest platform/cli/tests/ -v

# MCP adapter tests
.venv/bin/pytest platform/document-mcp/tests/ -v

# Coverage report
.venv/bin/pytest lib/tests/unit/ --cov=blue_data_doc_source --cov-report=term-missing

# Benchmarks (standalone — not pytest)
.venv/bin/python lib/tests/benchmarks/bench_chunkers.py
.venv/bin/python lib/tests/benchmarks/bench_retrieval.py
```
