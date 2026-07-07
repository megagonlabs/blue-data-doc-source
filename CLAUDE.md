# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Tests

```bash
# Unit tests (default)
.venv/bin/pytest lib/tests/unit/ -v

# Single file
.venv/bin/pytest lib/tests/unit/test_window_chunker.py -v

# Integration tests (requires Redis Stack on localhost:6379)
docker run -p 6379:6379 redis/redis-stack-server:latest
.venv/bin/pytest lib/tests/integration/ -v
```

## Running the example

```bash
python examples/ingest_txt.py sample.txt --embedding dummy
python examples/ingest_txt.py sample.txt --embedding openai
python examples/ingest_txt.py mydoc.pdf --embedding openai --query "eligibility requirements"
```

## Architecture

The pipeline is a strict one-way chain of single-responsibility layers:

```
File (.txt / .pdf)
  → parsers/      — extract raw text (TextParser) or page-by-page text (PDFParser)
  → chunkers/     — SectionChunker (default) detects headers and chunks within sections;
                    WindowChunker is the word-based sliding window fallback
  → embeddings/   — EmbeddingModel ABC with dim property;
                    OpenAIEmbeddingModel or DummyEmbeddingModel
  → chunk_record  — ChunkRecord dataclass: canonical chunk schema with provenance fields
  → enrichment/   — Enricher ABC; BasicEnricher does regex entity/keyword extraction
  → pipeline/     — DocumentPipeline orchestrates all layers
  → document_source/ — DocumentDataSource extends blue-data DataSource; public entry point
```

**`DocumentDataSource`** is the top-level public entry point. It extends blue-data's `DataSource`
abstraction and dispatches on `op` codes: `DOCUMENT.INGEST`, `DOCUMENT.SEARCH`,
`DOCUMENT.ENSURE_INDEX`, `DOCUMENT.DELETE`. It owns a single `DocumentPipeline` instance reused
across calls.

**`DocumentPipeline`** orchestrates the ingestion chain. Auto-detects `.txt` vs `.pdf` by
extension. `ingest(path)` returns `List[Dict]` and optionally upserts to a vector store.
`search(query)` embeds the query and delegates KNN to the injected `VectorSource`.
`ensure_index(collection)` calls `VectorSource.create_index` with the correct embedding dim.
`delete(path=...)` or `delete(document_id=..., chunk_count=...)` removes all chunks for a
document from the vector store.

**`VectorSource`** is blue-data's abstraction (lives in the `blue-data` repo). The pipeline
accepts any `VectorSource` implementation (e.g. `RedisVectorSource`) — no adapter layer needed.

**`ChunkRecord`** is the canonical schema shared across all document types. Key provenance
fields: `document_id`, `chunk_id`, `page_start`, `page_end`, `section_title`. These must
be preserved for downstream agent explainability.

## Key design constraints

- `document_id` is derived from a SHA-256 hash of the file's bytes — re-ingesting the same
  file produces the same chunk IDs, so vector store upsert overwrites rather than duplicates.
- PDF chunks are page-scoped: `page_start == page_end`. Chunks never span page boundaries in V1.
- `SectionChunker` is the default chunker. It falls back to window chunking when no section
  headers are detected. Pass a custom `chunker=` to `DocumentPipeline` to override.
- `WindowChunker` and `SectionChunker` both operate on **word count**, not character count.
- `EmbeddingModel` subclasses must implement a `dim` property. This is used by `ensure_index`
  and validated against the actual embedding length on every chunk.
- `DummyEmbeddingModel` is SHA-256 hash based — for testing only, not real similarity.
- No vector retrieval logic lives inside `DocumentPipeline` — it delegates entirely to `VectorSource`.
- `ChunkRecord.to_dict()` omits `None` optional fields (`page_start`, `page_end`, `section_title`,
  `embedding`) rather than serialising them as null.
- Call `DOCUMENT.ENSURE_INDEX` (or `pipeline.ensure_index()`) before the first ingest when
  using a vector store — the pipeline does not auto-create the index.
