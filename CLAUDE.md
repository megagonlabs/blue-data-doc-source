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
.venv/bin/pytest tests/ -v          # all tests
.venv/bin/pytest tests/test_window_chunker.py -v   # single file
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
  → chunkers/     — word-based sliding window (WindowChunker)
  → embeddings/   — EmbeddingModel ABC; OpenAIEmbeddingModel or DummyEmbeddingModel
  → models/       — ChunkRecord dataclass: canonical chunk schema with provenance fields
  → vector_stores/— VectorStore ABC; RedisVectorAdapter wraps an injected redis_source
  → pipeline/     — DocumentPipeline orchestrates all layers; entry point for callers
```

**`DocumentPipeline`** is the only public entry point. It auto-detects `.txt` vs `.pdf` by extension. `ingest(path)` returns `List[Dict]` and optionally upserts to a vector store. `search(query)` embeds the query and calls `VECTOR.KNN`.

**`RedisVectorAdapter`** does not own a Redis connection — callers inject a `redis_source` object that speaks `execute_query({op: ...})`. This adapter is a thin wrapper over Blue's existing `RedisVectorSource`, which lives in a separate repo.

**`ChunkRecord`** is the canonical schema shared across all document types. Key provenance fields: `document_id`, `chunk_id`, `page_start`, `page_end`, `section_title`. These must be preserved for downstream agent explainability.

## Key design constraints

- PDF chunks are page-scoped: `page_start == page_end`. Chunks never span page boundaries in V1.
- `WindowChunker` operates on **word count**, not character count.
- `DummyEmbeddingModel` is SHA-256 hash based — for testing only, not real similarity.
- No vector retrieval logic lives inside `DocumentPipeline` — it delegates entirely to `VectorStore`.
- `ChunkRecord.to_dict()` omits `None` optional fields (`page_start`, `page_end`, `section_title`, `embedding`) rather than serialising them as null.
