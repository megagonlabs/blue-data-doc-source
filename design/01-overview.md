# blue-data-doc-source — Overview

## Objective

AI agents that reason over enterprise documents fail when they cannot find the right passage, cite where it came from, or handle the variety of formats documents arrive in. A policy PDF, a Word contract, and a scanned insurance form all contain knowledge — but extracting it, making it searchable, and keeping it citable requires a pipeline that goes well beyond file reading.

blue-data-doc-source is that pipeline. It converts unstructured documents into structured, provenance-aware chunk records that AI agents can search, cite, and reason over. It plugs directly into blue-data's `DataSource` abstraction, making documents first-class members of the same data layer as SQL databases, graph stores, and JSON sources.

## What blue-data-doc-source Does

- **Parse** `.txt`, `.pdf`, and `.docx` files through a unified interface; fall back to Tesseract OCR for scanned pages with no embedded text layer
- **Classify** each document's type at ingest time (`KeywordClassifier` — zero API cost); propagate `document_type` to every chunk for pre-filtering in KNN search
- **Extract** typed schema fields from each document (`LLMSchemaExtractor` — one call per document): `parties`, `governing_law`, `total_due`, `effective_date`, and more; stored in a pluggable `SchemaStore`
- **Query** the schema store with structured filters (`DOCUMENT.FIND`): `{"fields.total_due": {"gt": 30000}}`, `{"fields.governing_law": {"contains": "California"}}`; no embedding, no LLM, deterministic; persists across restarts via `FileSchemaStore`
- **Chunk** extracted text into overlapping, size-bounded segments — section-aware by default so chunk boundaries respect document structure and every chunk carries its parent section heading
- **Enrich** each chunk with entities and semantic hints — `BasicEnricher` (regex, no ML dependency) or `LLMEnricher` (one call per chunk; keyphrases, relationships, semantic flags, language, summary; any provider via `llm_fn` injection)
- **Embed** chunks with any `EmbeddingModel` implementation — OpenAI out of the box, or a custom subclass for any provider
- **Store** embedded chunks in any blue-data `VectorSource` backend (in-memory for development, Redis Stack for production) via content-addressed upsert
- **Search** the vector store with a natural-language query: embed the query, run KNN, return ranked chunk records with full provenance; pre-filter by `document_type`, `document_id`, `section_title`, `content_type`, or page range before scoring
- **Deduplicate** automatically: `document_id` is a SHA-256 hash of the file bytes, so re-ingesting the same file overwrites rather than duplicates; changed files replace their old chunks
- **Expose** all operations through blue-data's `DataSource` interface (`DOCUMENT.INGEST`, `DOCUMENT.SEARCH`, `DOCUMENT.FIND`, `DOCUMENT.GET_SCHEMA`, `DOCUMENT.SEARCH_RICH`, `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.DELETE`) and optionally through a natural-language query interface backed by an LLM parser

## Design Philosophy

**Strict one-way pipeline** — the ingestion chain is a linear sequence of single-responsibility stages: parse → chunk → enrich → embed → store. No stage has knowledge of the stages before or after it. This makes each stage independently testable and independently replaceable.

**Every extension point is an ABC** — `Parser`, `Chunker`, `Enricher`, and `EmbeddingModel` are all abstract base classes. Built-in implementations cover the common cases; any of them can be replaced by a custom subclass injected at pipeline construction. The pipeline itself does not change.

**Content-addressed identity** — chunk IDs are derived from the document's SHA-256 hash, not from file paths or timestamps. Re-ingesting identical content is always a safe upsert. A changed file produces a new `document_id`, triggering deletion of its old chunks before the new ones are written.

**Provenance on every chunk** — `ChunkRecord` carries `file_name`, `page_start`, `page_end`, and `section_title` on every chunk. Agents can always answer "which document, which page, and which section does this passage come from?" without additional lookups.

**Blue-data VectorSource — no adapter needed** — the pipeline accepts any `VectorSource` implementation from blue-data directly. There is no intermediate adapter or doc-source-specific storage abstraction. Adding a new vector backend to blue-data makes it immediately available here.
