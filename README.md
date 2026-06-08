# blue-data-doc-source

Document ingestion, enrichment, and retrieval pipeline for Blue Data Intelligence. Parses `.txt` and `.pdf` files, chunks them, optionally enriches with entity/keyword extraction, embeds via OpenAI, and upserts to a Redis vector store.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Ingest a file with no embeddings — useful for inspecting chunks before wiring up a vector store:

```python
from blue_doc_source.pipeline import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("mydoc.pdf")

for r in records:
    print(r["chunk_id"], r["page_start"], r["text"][:80])
```

## End-to-end pipeline

Full pipeline: parse → chunk → enrich → embed → upsert to Redis → search.

```python
from blue_doc_source.source import DocumentDataSource
from blue_doc_source.embeddings import OpenAIEmbeddingModel
from blue_doc_source.enrichment import BasicEnricher
from blue_doc_source.vector_stores import RedisVectorAdapter

# Wrap Blue's existing RedisVectorSource
redis_source = ...  # your Blue RedisVectorSource instance
vector_store = RedisVectorAdapter(redis_source, default_dim=1536)

source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
    vector_store=vector_store,
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

### Without enrichment or a vector store

Embeddings and enrichment are both optional. Omit either to skip that step:

```python
source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),  # embeds but no storage
)
records = source.execute_query({"op": "DOCUMENT.INGEST", "path": "report.txt"})
```

Upsert only fires when both `embedding_model` and `vector_store` are provided and the chunk has an embedding.

## CLI example

```bash
# Ingest with dummy embeddings (no OpenAI key required)
python examples/ingest_txt.py sample.txt --embedding dummy

# Ingest a PDF with OpenAI embeddings and run a search query
python examples/ingest_txt.py report.pdf --embedding openai --query "eligibility requirements"
```

## Architecture

```
DocumentDataSource          ← public entry point (execute_query op-code interface)
  └─ DocumentPipeline       ← orchestrates all layers; auto-detects .txt vs .pdf
       ├─ TextParser         ← reads .txt → raw string
       ├─ PDFParser          ← reads .pdf → [{page_num, text}, ...] via pypdf
       ├─ WindowChunker      ← word-based sliding window → List[str]
       ├─ BasicEnricher      ← regex entity/keyword extraction (optional)
       ├─ OpenAIEmbeddingModel / DummyEmbeddingModel  (optional)
       ├─ ChunkRecord        ← canonical chunk schema with provenance fields
       └─ RedisVectorAdapter ← wraps Blue's RedisVectorSource (optional)
```

### ChunkRecord schema

Every ingested chunk becomes a `ChunkRecord`, serialised via `.to_dict()`. Optional fields are omitted when `None`.

| Field | Type | Notes |
|---|---|---|
| `document_id` | str | UUID shared across all chunks of the same file |
| `chunk_id` | str | `{document_id}_chunk_{index}` |
| `source_type` | str | Always `"document"` |
| `file_name` | str | Basename of the source file |
| `mime_type` | str | `"text/plain"` or `"application/pdf"` |
| `text` | str | Chunk text |
| `page_start` | int \| omitted | PDF only — 1-based page number |
| `page_end` | int \| omitted | PDF only — same as `page_start` in V1 |
| `section_title` | str \| omitted | Reserved for future use |
| `entities` | list | Extracted entities: `{text, type}` each |
| `semantic_hints` | dict | `word_count`, `sentence_count`, `has_questions`, `has_numbers`, `keywords` |
| `embedding` | list[float] \| omitted | Present only when an embedding model is set |
| `metadata` | dict | `source_path`, `chunk_index`, `chunk_size`, `overlap`, `ingested_at`, `embedding_model` |

### Chunking

`WindowChunker` uses **word count** (not character count). With `chunk_size=1000, overlap=150`, each chunk is at most 1000 words and consecutive chunks share 150 words of context. PDF chunks never span page boundaries.

### Enrichment

`BasicEnricher` is regex-based (no ML dependency). It extracts:
- **Entities:** `DATE`, `EMAIL`, `PERCENT`, `MONEY`, `PROPER_NOUN`
- **Semantic hints:** word/sentence count, top keywords (stopword-filtered), boolean flags

### Embeddings

| Class | Use case |
|---|---|
| `OpenAIEmbeddingModel` | Production; calls `text-embedding-3-small` by default |
| `DummyEmbeddingModel` | Tests; SHA-256 hash → fixed-dim float vector (not real similarity) |

### Vector store

`RedisVectorAdapter` does not own a Redis connection. Inject Blue's `RedisVectorSource` as `redis_source`; the adapter translates `upsert` / `search` calls to `execute_query({op: "VECTOR.*", ...})`.

## Running tests

```bash
.venv/bin/pytest tests/ -v
```
