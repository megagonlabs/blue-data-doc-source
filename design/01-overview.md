# blue-data-doc-source — Overview

## Objective

AI agents that reason over enterprise documents fail when they cannot find the right passage, cite where it came from, or handle the variety of formats documents arrive in. A policy PDF, a Word contract, and a scanned insurance form all contain knowledge — but extracting it, making it searchable, and keeping it citable requires a pipeline that goes well beyond file reading.

blue-data-doc-source is that pipeline. It converts unstructured documents into structured, provenance-aware chunk records that AI agents can search, cite, and reason over. It plugs directly into blue-data's `DataSource` abstraction, making documents first-class members of the same data layer as SQL databases, graph stores, and JSON sources.

## What blue-data-doc-source Does

- **Parse** `.txt`, `.pdf`, and `.docx` files through a unified interface; fall back to Tesseract OCR for scanned pages with no embedded text layer
- **Classify** each document's type at ingest time (`KeywordClassifier` — zero API cost); propagate `document_type` to every chunk for pre-filtering in KNN search; built-in types (`contract`, `invoice`, `policy`, `research_paper`, `financial_report`) plus user-defined types via `extra_types` — no subclassing required
- **Extract** typed fields from each document (`LLMFieldExtractor` — one call per document): `parties`, `governing_law`, `total_due`, `effective_date`, and more; user-defined field specs via `extra_types` for custom document types; stored in a pluggable `RecordStore`
- **Query** the schema store with structured filters (`DOCUMENT.FIND`): `{"fields.total_due": {"gt": 30000}}`, `{"fields.governing_law": {"contains": "California"}}`; no embedding, no LLM, deterministic; persists across restarts via `FileRecordStore`
- **Find and retrieve** in one call (`DOCUMENT.FIND_THEN_SEARCH`): filter documents by type or extracted fields, search text within each matched document, and return a merged ranked list — without the caller writing a loop
- **Chunk** extracted text into overlapping, size-bounded segments — section-aware by default so chunk boundaries respect document structure and every chunk carries its parent section heading
- **Enrich** each chunk with entities and semantic hints — `BasicEnricher` (regex, no ML dependency) or `LLMEnricher` (one call per chunk; keyphrases, relationships, semantic flags, language, summary; any provider via `llm_fn` injection)
- **Embed** chunks with any `EmbeddingModel` implementation — OpenAI out of the box, or a custom subclass for any provider
- **Store** embedded chunks in any blue-data `VectorSource` backend (in-memory for development, Redis Stack for production) via content-addressed upsert
- **Search** the vector store with a natural-language query: embed the query, run KNN, return ranked chunk records with full provenance; pre-filter by `document_type`, `document_id`, `section_title`, `content_type`, or page range before scoring
- **Deduplicate** automatically: `document_id` is a SHA-256 hash of the file bytes, so re-ingesting the same file overwrites rather than duplicates; changed files replace their old chunks
- **Graph-index** each document into a knowledge graph during ingest (optional): `Document` nodes (with `mime_type`, `ingested_at`, `chunk_count`), `Section` nodes, `Chunk` nodes, and `Entity` nodes extracted by `BasicEnricher`; edges: `HAS_SECTION`, `HAS_CHUNK`, `NEXT`, `PREV`, `MENTIONS`, `APPEARS_IN`, `CO_OCCURS_WITH`; all node IDs are deterministic so re-ingesting the same file is a no-op; entity nodes are shared across documents so "Acme Corp" is a single node reachable from any document that mentions it
- **Traverse** the graph by entity name (`DOCUMENT.GRAPH_NEIGHBORS` — deterministic node ID, no corpus scan, no embeddings) or by arbitrary traversal from any node (`DOCUMENT.GRAPH_QUERY` — start node, edge type, direction, limit); walk reading order via NEXT/PREV chains; ask which documents an entity appears in via the direct `APPEARS_IN` edge; inspect section hierarchy via `HAS_SECTION` → `HAS_CHUNK`; works with any blue-data `GraphSource` backend (`MemoryGraphSource` for dev, `NEO4JSource` for production)
- **Expose** all operations through blue-data's `DataSource` interface (`DOCUMENT.INGEST`, `DOCUMENT.SEARCH`, `DOCUMENT.FIND`, `DOCUMENT.FIND_THEN_SEARCH`, `DOCUMENT.GET_RECORD`, `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.DELETE`, `DOCUMENT.GRAPH_NEIGHBORS`, `DOCUMENT.GRAPH_QUERY`) and optionally through a natural-language query interface (`NLQueryParser`) that maps plain-English instructions to any of these ops automatically; `DOCUMENT.SEARCH` accepts `"rich": true` to return provenance, neighbor context, and by_document grouping instead of a flat list

## Design Philosophy

**Strict one-way pipeline** — the ingestion chain is a linear sequence of single-responsibility stages: parse → chunk → enrich → embed → store. No stage has knowledge of the stages before or after it. This makes each stage independently testable and independently replaceable.

**Every extension point is an ABC** — `Parser`, `Chunker`, `Enricher`, and `EmbeddingModel` are all abstract base classes. Built-in implementations cover the common cases; any of them can be replaced by a custom subclass injected at pipeline construction. The pipeline itself does not change.

**Content-addressed identity** — chunk IDs are derived from the document's SHA-256 hash, not from file paths or timestamps. Re-ingesting identical content is always a safe upsert. A changed file produces a new `document_id`, triggering deletion of its old chunks before the new ones are written.

**Provenance on every chunk** — `ChunkRecord` carries `file_name`, `page_start`, `page_end`, and `section_title` on every chunk. Agents can always answer "which document, which page, and which section does this passage come from?" without additional lookups.

**Blue-data VectorSource — no adapter needed** — the pipeline accepts any `VectorSource` implementation from blue-data directly. There is no intermediate adapter or doc-source-specific storage abstraction. Adding a new vector backend to blue-data makes it immediately available here.

**Blue-data GraphSource — same principle** — the graph layer accepts any `GraphSource` from blue-data directly. `MemoryGraphSource` for dev, `NEO4JSource` for production — no adapter, no extra configuration. The same `GraphIndexer` writes identical node IDs and edge types regardless of the backend. Deterministic node IDs mean re-ingesting the same file is a no-op for the graph (Document node already exists → skip).
