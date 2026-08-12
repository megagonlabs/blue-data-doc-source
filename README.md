# blue-data-doc-source

## Overview

blue-data-doc-source is a document ingestion and retrieval source for Blue Data Intelligence. It sits upstream of Blue's vector layer, converting unstructured documents into structured, provenance-aware chunk records that AI agents can search, cite, and reason over.

It handles the full pipeline from raw file to queryable vector index: parsing text, PDFs, and DOCX files, detecting section structure, extracting entities and keywords, generating embeddings, and upserting into Blue's RedisVectorSource. It plugs directly into Blue's DataSource abstraction, so documents become first-class members of the same data layer as SQL databases, graph stores, and JSON sources.

It answers questions like: what does this document say, which section does it come from, what page is it on, and how does it relate to a user's query.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Ingest a file with no embeddings — useful for inspecting chunks before wiring up a vector store:

```python
from blue_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("mydoc.pdf")

for r in records:
    print(r["chunk_id"], r["page_start"], r["text"][:80])
```

## End-to-end pipeline

Full pipeline: parse → chunk → enrich → embed → upsert to Redis → search.

`blue-data-doc-source` depends on `blue-data`. Pass a `VectorSource` instance (e.g. `RedisVectorSource`) from blue-data as the `vector_source`.

```python
from blue_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

# Instantiate blue-data's RedisVectorSource
redis_source = RedisVectorSource(name="my_redis", host="localhost", port=6379)

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
    vector_source=redis_source,
    enricher=BasicEnricher(),
    chunk_size=1000,
    overlap=150,
    default_collection="my_docs",
)

# Ingest a document
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
for r in results:
    print(r["chunk_id"], r["text"][:120])
```

### Without enrichment or a vector source

Embeddings and enrichment are both optional. Omit either to skip that step:

```python
source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),  # embeds but no storage
)
records = source.execute_query({"op": "DOCUMENT.INGEST", "path": "report.txt"})
```

Upsert only fires when both `embedding_model` and `vector_source` are provided and the chunk has an embedding.

## CLI example

```bash
# Ingest the sample policy — no embeddings, just inspect chunks
python examples/ingest_txt.py

# Ingest with dummy embeddings
python examples/ingest_txt.py --embedding dummy

# Ingest your own file
python examples/ingest_txt.py path/to/doc.txt --embedding dummy --query "eligibility requirements"
```

## Local end-to-end demo (PDF → Redis → search)

`examples/demo_pdf.py` runs the full pipeline against a local Redis Stack instance.
Sample data is included in `data/` — no file needed to get started.

### Prerequisites

**Redis Stack** (provides RediSearch + RedisJSON):

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

> If Redis is already running on port 6379 (e.g. from a previous session), this command will fail — that's fine, just skip it and proceed.

**Install dependencies** (includes `redis[hiredis]`):

```bash
pip install -e .
```

### Run

```bash
# Zero-config — uses data/sample_policy.pdf and a built-in query
python examples/demo_pdf.py

# With OpenAI embeddings for real semantic search
OPENAI_API_KEY=sk-... python examples/demo_pdf.py \
    --embedding openai \
    --query "what is the deductible"

# Your own PDF
OPENAI_API_KEY=sk-... python examples/demo_pdf.py path/to/doc.pdf \
    --embedding openai \
    --query "termination clause"
```

### Natural language queries

`--nl-query` lets you ask in plain English. Under the hood, `NLQueryParser` uses an LLM (GPT-4o-mini by default) to parse the instruction into the right op and parameters, then calls `execute_query` automatically.

```python
from blue_doc_source import DocumentDataSource, NLQueryParser, OpenAIEmbeddingModel

source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=redis_source,
    nl_parser=NLQueryParser(),
)

results = source.execute_query_nl("find the termination clause")
records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")
```

You can swap in any LLM backend via `parse_fn`:

```python
parser = NLQueryParser(parse_fn=lambda text: my_llm_call(text))
```

### Options

| Flag | Default | Description |
|---|---|---|
| `path` | `data/sample_policy.pdf` | Path to a `.pdf` file |
| `--query` | built-in | Search query to run after ingestion |
| `--nl-query` | — | Natural language query — LLM parses intent automatically (requires `--embedding openai`) |
| `--embedding` | `dummy` | `dummy` (SHA-256 hash) or `openai` |
| `--openai-model` | `text-embedding-3-small` | OpenAI embedding model |
| `--chunk-size` | `300` | Max words per chunk |
| `--overlap` | `50` | Word overlap between chunks |
| `--top-k` | `3` | Number of results to return |
| `--collection` | `demo_docs` | Redis key prefix and index name |
| `--redis-host` | `localhost` | Redis host |
| `--redis-port` | `6379` | Redis port |

> **Note:** `examples/local_redis_store.py` is demo-only and is not part of the `blue_doc_source` package. Production use should pass Blue's `RedisVectorSource` directly as `vector_source`.

## Architecture

```
DocumentDataSource (extends blue-data DataSource)  ← public entry point
  └─ DocumentPipeline       ← orchestrates all layers; auto-detects .txt vs .pdf
       ├─ TextParser         ← reads .txt → raw string
       ├─ PDFParser          ← reads .pdf → [{page_num, text}, ...] via pypdf; OCR fallback via Tesseract
       ├─ DocxParser         ← reads .docx → markdown-formatted string via python-docx
       ├─ SectionChunker     ← detects headers, chunks within sections (default)
       │    └─ WindowChunker ← word-based sliding window fallback
       ├─ BasicEnricher      ← regex entity/keyword extraction (optional)
       ├─ OpenAIEmbeddingModel / DummyEmbeddingModel  (optional)
       ├─ ChunkRecord        ← canonical chunk schema with provenance fields
       └─ VectorSource       ← blue-data VectorSource injected by caller (optional)
```

### ChunkRecord schema

Every ingested chunk becomes a `ChunkRecord`, serialised via `.to_dict()`. Optional fields are omitted when `None`.

| Field | Type | Notes |
|---|---|---|
| `document_id` | str | SHA-256 content hash of the source file — stable across re-ingests |
| `chunk_id` | str | `{document_id}_chunk_{index}` |
| `source_type` | str | Always `"document"` |
| `file_name` | str | Basename of the source file |
| `mime_type` | str | `"text/plain"`, `"application/pdf"`, or `"application/vnd.openxmlformats-officedocument.wordprocessingml.document"` |
| `text` | str | Chunk text |
| `page_start` | int \| omitted | PDF only — 1-based page number |
| `page_end` | int \| omitted | PDF only — same as `page_start` in V1 |
| `section_title` | str \| omitted | Section heading detected by SectionChunker — omitted when no structure found |
| `entities` | list | Extracted entities: `{text, type}` each |
| `semantic_hints` | dict | `word_count`, `sentence_count`, `has_questions`, `has_numbers`, `keywords` |
| `embedding` | list[float] \| omitted | Present only when an embedding model is set |
| `metadata` | dict | `source_path`, `chunk_index`, `chunk_size`, `overlap`, `ingested_at`, `embedding_model` |

### Chunking

`SectionChunker` (the default) detects section headers — markdown (`# Title`), numbered (`1.2 Overview`), ALL-CAPS, and underlined (`Title\n===`) — and chunks within each section. The detected header becomes `section_title` on each chunk. When no headers are found it falls back to plain window chunking with `section_title=None`.

`WindowChunker` uses **word count** (not character count). With `chunk_size=1000, overlap=150`, each chunk is at most 1000 words and consecutive chunks share 150 words of context. PDF chunks never span page boundaries.

Pass a custom chunker to override the default:

```python
from blue_doc_source.source.document.chunkers import WindowChunker
pipeline = DocumentPipeline(chunker=WindowChunker(chunk_size=500, overlap=50))
```

### Enrichment

`BasicEnricher` is regex-based (no ML dependency). It extracts:
- **Entities:** `DATE`, `EMAIL`, `PERCENT`, `MONEY`, `PROPER_NOUN`
- **Semantic hints:** word/sentence count, top keywords (stopword-filtered), boolean flags

### Embeddings

| Class | Use case |
|---|---|
| `OpenAIEmbeddingModel` | Production; calls `text-embedding-3-small` by default (1536-dim) |
| `DummyEmbeddingModel` | Tests; SHA-256 hash → fixed-dim float vector (not real similarity) |

All embedding models expose a `dim` property used by `ensure_index` and validated against the actual embedding length on every chunk. `OpenAIEmbeddingModel` supports `text-embedding-3-small` (1536), `text-embedding-3-large` (3072), and `text-embedding-ada-002` (1536).

### Pluggable parsers

The built-in parsers (`TextParser`, `PDFParser`) cover standard digitally-created files. To use an external document processor — LlamaIndex, a vision model, a cloud OCR API — implement the `Parser` ABC and pass it as `parser=` to `DocumentPipeline`. Everything downstream (chunking, enrichment, embedding, storage) is unchanged.

```python
from blue_doc_source import DocumentPipeline, Parser

class LlamaIndexParser(Parser):
    def parse(self, path: str):
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return [{"text": doc.text} for doc in docs]

pipeline = DocumentPipeline(parser=LlamaIndexParser(), embedding_model=...)
pipeline.ingest("complex_layout.pdf")  # LlamaIndex parses; our pipeline chunks and stores
```

`parse()` must return a list of dicts with `"text": str`. Include `"page_num": int` (1-based) for page-aware documents — it propagates to `page_start` / `page_end` on each `ChunkRecord`.

#### Bypassing the parser entirely

When you already have extracted text or pages from an external tool, skip file parsing altogether:

```python
# Pre-extracted text (no file path needed)
pipeline.ingest_text(text, file_name="report.txt")

# Pre-extracted pages from a vision model
pipeline.ingest_pages(
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}],
    file_name="contract.pdf",
)
```

`document_id` is derived from a SHA-256 hash of the content, so calling these methods twice with the same text is safe — it overwrites rather than duplicates.

### Scanned PDF support (OCR)

By default, `PDFParser` uses pypdf to read the text layer embedded in a PDF. Pages with no text layer — scanned documents, photographed pages — are silently skipped.

Pass `ocr_fallback=True` to run Tesseract OCR on any blank page instead of dropping it:

```python
from blue_doc_source.source.document.parsers import PDFParser

parser = PDFParser(ocr_fallback=True)
pages = parser.parse_pages("scanned_contract.pdf")
```

OCR'd pages carry `"ocr": True` in the result dict so downstream code can identify them.

**Install OCR dependencies:**

```bash
pip install blue-data-doc-source[ocr]   # installs pytesseract + pdf2image
brew install tesseract                   # macOS — Tesseract must be on PATH
```

> `pdf2image` also requires [Poppler](https://poppler.freedesktop.org/). On macOS: `brew install poppler`. On Ubuntu: `apt-get install poppler-utils`.

A clear `ImportError` is raised if OCR is requested but the dependencies are not installed.

### Vector source and index creation

`DocumentDataSource` and `DocumentPipeline` accept any `VectorSource` from blue-data (e.g. `RedisVectorSource`, `MemoryVectorSource`). The pipeline calls `.upsert()` on ingest and `.knn()` on search — no adapter layer needed.

Before the first ingest, create the vector index using `DOCUMENT.ENSURE_INDEX`:

```python
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})
```

Or directly on the pipeline:

```python
pipeline.ensure_index(collection="contracts")
```

This calls `VectorSource.create_index` with the correct embedding dimension. If the index already exists the behaviour depends on the backend (Redis raises an error; call it once at startup).

## Running tests

```bash
# Unit tests (no external services required)
.venv/bin/pytest lib/tests/unit/ -v

# Integration tests (requires Redis Stack on localhost:6379)
docker run -p 6379:6379 redis/redis-stack-server:latest
.venv/bin/pytest lib/tests/integration/ -v
```
