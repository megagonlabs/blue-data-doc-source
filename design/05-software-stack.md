# blue-data-doc-source — Software Stack

---

## Language & Runtime

| Concern | Choice |
|---------|--------|
| Language | Python |
| Minimum version | 3.10 |
| Package manager | pip |
| Project config | `pyproject.toml` (root-level; `where = ["lib/src"]`) |

---

## Core Dependencies (always installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `blue-data` | local editable | `DataSource`, `VectorSource`, `GraphSource` ABCs and all backends |
| `openai` | ≥1.0.0 | `OpenAIEmbeddingModel`; `LLMFieldExtractor`; `LLMEnricher`; `StructuredFieldExtractor` |
| `pypdf` | ≥4.0.0 | `PDFParser` — text extraction from digital PDFs |
| `python-docx` | ≥1.0.0 | `DocxParser` — heading-preserving DOCX extraction |
| `redis[hiredis]` | ≥4.5.0 | `RedisVectorSource` and `RedisJSONSource` from `blue-data` (production backends) |

`openai` and `redis` are hard dependencies — they are always installed with the package. This is intentional: production deployments need both, and keeping them hard dependencies avoids version conflicts across optional extras.

---

## Optional Extras

### `[ocr]` — scanned PDF support

| Package | Purpose |
|---------|---------|
| `pytesseract` ≥0.3.10 | Python wrapper for Tesseract OCR |
| `pdf2image` ≥1.16.0 | Converts PDF pages to PIL images for Tesseract input |

`PDFParser` deferred-imports `pytesseract` and `pdf2image` inside its OCR fallback branch. The parser works without this extra for digital PDFs — it only falls back to OCR when `pypdf` returns an empty or near-empty text extraction.

### `[nlp]` — sentence-boundary chunking

| Package | Purpose |
|---------|---------|
| `nltk` ≥3.8 | Sentence tokenizer used by `SentenceChunker` |

`SentenceChunker` deferred-imports `nltk.tokenize.sent_tokenize`. The default chunker (`SectionChunker`) does not use `nltk`.

### `[mcp]` — MCP server

| Package | Purpose |
|---------|---------|
| `mcp` ≥1.0.0 | fastmcp server runtime for `platform/doc-mcp/src/server.py` |

---

## Parsers

| Parser | Package | Format | Notes |
|--------|---------|--------|-------|
| `TextParser` | stdlib | `.txt` | Reads UTF-8; no external dependency |
| `PDFParser` | `pypdf` | `.pdf` | Text extraction; falls back to `pytesseract` + `pdf2image` for scanned pages (`[ocr]`) |
| `PlumberPDFParser` | `pdfplumber` (not declared; caller installs) | `.pdf` | Table-aware; returns typed `parse_blocks()` output with `"type": "table"` blocks |
| `DocxParser` | `python-docx` | `.docx` | Preserves `Heading 1/2/3` structure for `SectionChunker` |

**Custom parser contract:** Implement `Parser.parse(path) -> list[{"text": str, "page_num"?: int}]`. Override `parse_blocks()` to return typed content blocks (prose + tables). Pass the instance as `parser=` to `DocumentPipeline`.

---

## Chunkers

All chunkers are implemented in pure Python — no library dependencies. Both `SectionChunker` and `WindowChunker` operate on **word count**, not character count.

| Chunker | Default? | Strategy |
|---------|----------|---------|
| `SectionChunker` | Yes | Detects Markdown/heading markers; groups content by section; falls back to window chunking when no structure detected |
| `WindowChunker` | No | Sliding word-count window with configurable `chunk_size` and `overlap` |
| `SentenceChunker` | No | Sentence-boundary aware via `nltk.sent_tokenize` (`[nlp]` extra required) |

---

## Embeddings

| Model | Package | Dimension | Notes |
|-------|---------|-----------|-------|
| `OpenAIEmbeddingModel` | `openai` | 1536 | `text-embedding-3-small`; API call per chunk |
| `DummyEmbeddingModel` | stdlib | configurable | SHA-256 hash of text → deterministic fixed vector; for tests |

`EmbeddingModel` ABC requires a `dim` property. `DocumentPipeline` calls `ensure_index(collection, dim=model.dim)` to validate alignment between the model dimension and the vector index configuration.

---

## Enrichment

| Enricher | Dependencies | What it produces |
|----------|-------------|-----------------|
| `BasicEnricher` | stdlib (regex) | `entities`: PROPER_NOUN matches (2+ title-case words); `semantic_hints`: `has_numeric`, `has_date`, `is_short` |
| `LLMEnricher` | `openai` | `entities`: typed entity list; keyphrases, relationships, semantic flags — one GPT-4o-mini call per chunk |

`Enricher.enrich(text)` must return `{"entities": list[{text, type}], "semantic_hints": dict}`.

---

## Schema Classification and Extraction

| Component | Zero-cost? | Notes |
|-----------|-----------|-------|
| `KeywordClassifier` | Yes | Scores first 500 words against signal phrase sets; returns `(document_type, confidence)` |
| `LLMFieldExtractor` | No | One GPT-4o-mini call per document; extracts type-specific fields (parties, dates, totals, etc.) |
| `StructuredFieldExtractor` | No | Caller supplies a Pydantic model, TypedDict, or JSON Schema; prompt auto-generated |

`DocumentRecord` schema: `{document_id, document_type, confidence, classifier, fields}`.

---

## Storage

### VectorSource (search index)

Provided by `blue-data`. `DocumentPipeline` interacts only with the `VectorSource` interface.

| Backend | Class | Persistence | Notes |
|---------|-------|-------------|-------|
| Memory | `MemoryVectorSource` | None | Dev/test; numpy cosine |
| Redis | `RedisVectorSource` | Yes | Production; requires Redis Stack |
| ChromaDB | `ChromaDBVectorSource` | Yes | Optional |

The pipeline uses six extra index fields for pre-filtering: `document_id`, `file_name`, `section_title`, `mime_type`, `page_start`, `content_type`. These are declared in `pipeline._INDEX_EXTRA_FIELDS` and passed to `create_index` on `DOCUMENT.ENSURE_INDEX`.

### GraphSource (entity / structure / provenance graph)

Also provided by `blue-data`. `graph_source=None` (the default) disables graph indexing with zero overhead.

| Backend | Class | Notes |
|---------|-------|-------|
| Memory | `MemoryGraphSource` | Dev/test |
| Neo4j | `Neo4JGraphSource` | Production |

### RecordStore (schema persistence)

| Class | Backed by | Notes |
|-------|-----------|-------|
| `MemoryRecordStore` | `MemoryJSONSource` | Default; in-process |
| `FileRecordStore` | `FileJSONSource` | Atomic write; survives restarts |
| `JsonSourceRecordStore` | Any `JSONSource` | Accepts `RedisJSONSource` for multi-process-safe production use |

---

## External Interfaces

### MCP Server (`platform/doc-mcp/`)

| Concern | Choice |
|---------|--------|
| Library | fastmcp (via `[mcp]` extra) |
| Transport | stdio |

Seven tools: `ensure_index`, `ingest_document`, `search_documents`, `find_documents`, `find_then_search`, `get_document_record`, `delete_document`.

### REST API (`platform/doc-api/`)

| Concern | Choice |
|---------|--------|
| Framework | FastAPI |
| ASGI server | Uvicorn |
| Port | 8001 |

Routes under `/documents/` mirror the seven ops. Request/response models are Pydantic `BaseModel`. No business logic in handlers — all calls delegate to the `DocumentPipeline` singleton.

### CLI (`platform/doc-cli/`)

| Concern | Choice |
|---------|--------|
| Argument parsing | Click |
| Output | plain text (default) or JSON (`--json` flag) |

Commands: `ensure-index`, `ingest`, `search`, `delete`.

---

## Testing

| Concern | Choice |
|---------|--------|
| Test runner | `pytest` |
| Default scope | `testpaths = ["lib/tests/unit"]` |
| Coverage | `pytest-cov` |

Unit tests use `DummyEmbeddingModel`, `MemoryVectorSource`, `MemoryGraphSource`, `MemoryRecordStore`, and `MagicMock` for `openai.OpenAI`. Integration tests (`lib/tests/integration/`) require Redis Stack on `localhost:6379` and are run explicitly.

---

## Dependency Summary

| Package | Version | Extra | Role |
|---------|---------|-------|------|
| `blue-data` | local | always | DataSource / VectorSource / GraphSource ABCs and backends |
| `openai` | ≥1.0.0 | always | Embeddings (OpenAIEmbeddingModel); LLMFieldExtractor; LLMEnricher; StructuredFieldExtractor |
| `pypdf` | ≥4.0.0 | always | PDFParser digital text extraction |
| `python-docx` | ≥1.0.0 | always | DocxParser DOCX extraction |
| `redis[hiredis]` | ≥4.5.0 | always | RedisVectorSource + RedisJSONSource production backends |
| `pytesseract` | ≥0.3.10 | `[ocr]` | OCR fallback in PDFParser for scanned pages |
| `pdf2image` | ≥1.16.0 | `[ocr]` | PDF page → PIL image for pytesseract |
| `nltk` | ≥3.8 | `[nlp]` | Sentence tokenizer for SentenceChunker |
| `mcp` | ≥1.0.0 | `[mcp]` | fastmcp runtime for document-mcp server |
| `fastapi` | latest | `platform/doc-api` | REST API framework |
| `uvicorn` | latest | `platform/doc-api` | ASGI server (port 8001) |
| `click` | latest | `platform/doc-cli` | CLI argument parsing |
| `pytest` | latest | dev | Test runner |
| `pytest-cov` | latest | dev | Coverage measurement |
