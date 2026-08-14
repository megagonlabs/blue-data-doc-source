# Developer Guide

## Quick start

Parse and chunk a file with no embeddings — useful for inspecting chunk structure before wiring up a vector store:

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("policy.pdf")

for r in records:
    print(r["chunk_id"], r.get("section_title"), r["text"][:80])
```

`DocumentPipeline` auto-detects the parser by extension: `.txt` → `TextParser`, `.pdf` → `PDFParser`, `.docx` → `DocxParser`.

---

## Full pipeline

The complete ingestion chain: parse → chunk → enrich → embed → upsert to vector store → search.

```python
from blue_data_doc_source import DocumentPipeline, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

pipeline = DocumentPipeline(
    chunk_size=500,
    overlap=100,
    enricher=BasicEnricher(),
    embedding_model=OpenAIEmbeddingModel(),   # requires OPENAI_API_KEY
    vector_source=MemoryVectorSource(),
    collection="contracts",
)

# Create the vector index before the first ingest
pipeline.ensure_index(collection="contracts")

# Ingest documents
pipeline.ingest("contract_a.pdf")
pipeline.ingest("contract_b.pdf")

# Search
results = pipeline.search("termination clause", k=5)
for r in results:
    print(r["chunk_id"], r["text"][:120])
```

Upsert fires only when both `embedding_model` and `vector_source` are provided and the chunk has an embedding. Omit either to skip persistence and just return records.

---

## DocumentDataSource

`DocumentDataSource` wraps `DocumentPipeline` behind blue-data's `DataSource` interface, exposing ops as a dict-based `execute_query` call. Use this when integrating with `DataRegistry` or the MCP server.

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    enricher=BasicEnricher(),
    chunk_size=500,
    overlap=100,
    default_collection="contracts",
)

# Create index
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})

# Ingest
records = source.execute_query({
    "op": "DOCUMENT.INGEST",
    "path": "contract.pdf",
    "collection": "contracts",
})
print(f"Ingested {len(records)} chunks")

# Search
results = source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "k": 5,
})
```

### Supported ops

| Op | Required fields | Returns |
|---|---|---|
| `DOCUMENT.INGEST` | `path` | `list[dict]` — one dict per chunk |
| `DOCUMENT.SEARCH` | `query` | `list[dict]` — ranked matching chunks |
| `DOCUMENT.ENSURE_INDEX` | — | `None` |
| `DOCUMENT.DELETE` | `path` OR (`document_id` + `chunk_count`) | `int` — deleted chunk count |

---

## Registering with DataRegistry

`DocumentDataSource` extends blue-data's `DataSource`, so it can be registered alongside SQL and graph sources in a single `DataRegistry`:

```python
from blue_data.registry.data_registry import DataRegistry
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel

registry = DataRegistry()
doc_source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=my_vector_source,
)
registry.register_source(doc_source)

# Ingest via registry
registry.execute_query(
    source="contracts",
    query={"op": "DOCUMENT.INGEST", "path": "contract.pdf"},
)
```

---

## Chunking

The default chunker is `SectionChunker`. It detects section headers (markdown, numbered, ALL-CAPS, underlined) and chunks within each section, preserving the heading as `section_title` on every chunk. Falls back to word-count windowing when no headers are found.

```python
# Default: SectionChunker
pipeline = DocumentPipeline(chunk_size=500, overlap=100)

# Override with WindowChunker for unstructured prose
from blue_data_doc_source import WindowChunker
pipeline = DocumentPipeline(chunker=WindowChunker(chunk_size=300, overlap=50))

# SentenceChunker — no sentence ever split across chunks (requires [nlp] extra for best results)
from blue_data_doc_source import SentenceChunker
pipeline = DocumentPipeline(chunker=SentenceChunker(chunk_size=400, overlap=60))
```

All chunkers measure size in **words**, not characters or tokens. See [chunkers.md](chunkers.md) for tuning guidance and custom chunker implementation.

---

## Enrichment

`BasicEnricher` is regex-based — no ML dependencies. It extracts named entities and computes semantic hints, both stored on each `ChunkRecord`.

```python
from blue_data_doc_source import BasicEnricher

pipeline = DocumentPipeline(enricher=BasicEnricher(), chunk_size=500, overlap=100)
records = pipeline.ingest("policy.pdf")

for r in records:
    print(r["entities"])        # [{"text": "January 15, 2026", "type": "DATE"}, ...]
    print(r["semantic_hints"])  # {"word_count": 312, "keywords": [...], ...}
```

Enrichment is optional — omit `enricher` and `entities` / `semantic_hints` are empty on every record. See [extending.md](extending.md) to implement a custom enricher.

---

## Embeddings

| Class | Use | Notes |
|---|---|---|
| `OpenAIEmbeddingModel` | Production | Requires `OPENAI_API_KEY`; default model `text-embedding-3-small` (1536-dim) |
| `DummyEmbeddingModel` | Tests | SHA-256 hash → fixed-dim float vector; not real similarity |

```python
from blue_data_doc_source import OpenAIEmbeddingModel, DummyEmbeddingModel

# Production
model = OpenAIEmbeddingModel(model="text-embedding-3-small")  # 1536-dim

# Testing (no API key needed)
model = DummyEmbeddingModel(dim=8)
```

All embedding models expose a `dim` property used by `ensure_index` and validated against the actual embedding length on every chunk. See [extending.md](extending.md) to implement a custom embedding model.

---

## Vector storage

Pass any `VectorSource` from blue-data as `vector_source`:

```python
# In-memory — development and testing
from blue_data.source.vector.memory_vector_source import MemoryVectorSource
vs = MemoryVectorSource()

# Redis Stack — production (requires redis-stack-server)
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
vs = RedisVectorSource(name="docs", host="localhost", port=6379)
```

Always call `ensure_index` (or `DOCUMENT.ENSURE_INDEX`) before the first ingest. The pipeline does not auto-create the index.

---

## Search

`DocumentPipeline.search` embeds the query and runs KNN against the vector store:

```python
results = pipeline.search("payment obligations", k=5, collection="contracts")
for r in results:
    print(r["chunk_id"], r.get("section_title"), r["text"][:120])
```

`DocumentDataSource.execute_query` with `DOCUMENT.SEARCH` delegates to the same path.

### Pre-filtering

Pass a `filters` dict to scope the KNN search before it runs. Pre-filtering is applied at the vector store level — only the matching subset of chunks is scored.

```python
# Scope to a single document
results = pipeline.search(
    "termination clause",
    filters={"document_id": "doc_3a7f1c2e"},
)

# Scope to a file type
results = pipeline.search(
    "revenue figures",
    filters={"mime_type": "application/pdf"},
)

# Scope to a specific file
results = pipeline.search(
    "eligibility",
    filters={"file_name": "policy_2024.pdf"},
)

# Scope to a section
results = pipeline.search(
    "notice period",
    filters={"section_title": "Termination"},
)

# Page range (PDFs only — chunks without page_start are excluded)
results = pipeline.search(
    "definitions",
    filters={"page_start": {"gte": 1, "lte": 10}},
)

# Exact page
results = pipeline.search("liability cap", filters={"page_start": 5})

# Combine filters (AND)
results = pipeline.search(
    "payment terms",
    filters={"file_name": "contract_a.pdf", "section_title": "Pricing"},
)
```

Via the op interface:

```python
source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "filter": {"document_id": "doc_3a7f1c2e"},
})
```

**Supported filter keys:**

| Key | Type | Match |
|-----|------|-------|
| `document_id` | str | Exact — scope to one document |
| `file_name` | str | Exact — scope to one file |
| `section_title` | str | Exact — scope to one section heading |
| `mime_type` | str | Exact — scope to a file type |
| `page_start` | int or `{"gte": N, "lte": M}` | Numeric range — PDF chunks only |

All filters are combined with AND. `None` values are ignored. Chunks where an optional field is absent (e.g. `page_start` on a TXT chunk) are excluded when that field is filtered.

**Index requirement:** pre-filtering requires an index created after this feature was added. Existing Redis indexes without the filterable fields must be dropped and recreated with `ensure_index()`.

---

## Natural language queries

`NLQueryParser` uses an LLM to translate plain-English instructions into `DOCUMENT.*` op dicts, then executes them via `execute_query`.

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=my_vector_source,
    nl_parser=NLQueryParser(),   # uses GPT-4o-mini by default; requires OPENAI_API_KEY
)

results = source.execute_query_nl("find the termination clause in the contracts collection")
records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")
```

Inject a custom LLM via `parse_fn`:

```python
parser = NLQueryParser(parse_fn=lambda text: my_llm_call(text))
```

---

## Deduplication and re-ingestion

`document_id` is derived from a SHA-256 hash of the file's bytes. Re-ingesting the same file produces the same `document_id` and the same `chunk_id` values — vector store upsert overwrites rather than duplicates.

When a file's content changes, the pipeline detects the new `document_id`, deletes the old chunks from the vector store, and upserts the new ones. This cleanup is **in-process only** — stale chunks from a previous process run must be removed explicitly:

```python
# Using path (requires same pipeline instance that did the ingest)
pipeline.delete(path="contract.pdf", collection="contracts")

# Using document_id + chunk_count (cross-process safe)
pipeline.delete(document_id="doc_abc123", chunk_count=14, collection="contracts")

# Via DocumentDataSource
source.execute_query({
    "op": "DOCUMENT.DELETE",
    "document_id": "doc_abc123",
    "chunk_count": 14,
    "collection": "contracts",
})
```

---

## Pre-extracted text and pages

Skip file parsing entirely when text has already been extracted by an external tool:

```python
# Full-document text
pipeline.ingest_text(text, file_name="report.txt", collection="reports")

# Page-level output (page_num propagates to ChunkRecord.page_start / page_end)
pipeline.ingest_pages(
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}],
    file_name="scanned.pdf",
    collection="reports",
)
```

---

## Scanned PDFs (OCR)

Enable Tesseract OCR for pages with no embedded text layer:

```python
pipeline = DocumentPipeline(ocr_fallback=True, ...)
pipeline.ingest("scanned_contract.pdf")
```

Requires the `[ocr]` extra and Tesseract on PATH. See [installation.md](installation.md) for setup instructions. OCR pages carry `metadata["ocr"] = True` and `metadata["ocr_confidence"]` on each chunk record.
