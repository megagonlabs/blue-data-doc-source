# blue-data-doc-source — Decision Log

Each entry records a significant design decision: what was decided, why, and what consequences follow.

---

## ADR-001 — SHA-256 of file bytes as document ID

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `document_id` is the SHA-256 hex digest of the file's raw bytes. It is computed before any parsing.

**Rationale:** Content-addressable IDs make re-ingest idempotent. The same file always produces the same `document_id` and the same `chunk_id`s (`{document_id}_chunk_{index:04d}`). Vector store upsert overwrites existing vectors rather than creating duplicates. Graph indexing detects an existing `Document` node and skips re-creation. No external ID registry is needed.

**Consequences:**
- Changing a file's content (even one byte) produces a new `document_id`. The old chunks remain in the vector store until explicitly deleted.
- File rename does not change `document_id`. Two differently-named files with the same content produce the same ID — intended.
- Callers must call `DOCUMENT.DELETE` on the old ID before re-ingesting a changed file if they want to remove stale chunks.

---

## ADR-002 — Synchronous pipeline (no async)

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `DocumentPipeline` and all its stages (parsers, chunkers, enrichers, embedding models) are synchronous. There is no `async`/`await` in the library layer.

**Rationale:** Document ingestion is CPU-bound (parsing, chunking) and network-bound (one embedding call per chunk, one enrichment call per chunk). Making it async adds complexity without meaningful throughput gain for single-document workloads. The REST API layer handles concurrency via Uvicorn workers; the library does not need to.

**Consequences:**
- FastAPI route handlers block the event loop during ingest. For production with many concurrent ingests, run multiple Uvicorn workers or offload to a task queue.
- Adding true async pipeline support in the future is possible — all ABCs would need `async` method variants.

---

## ADR-003 — Word-count chunking (not character count)

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `SectionChunker` and `WindowChunker` both measure chunk size in **word count**, not character count. `chunk_size=1000` means 1000 words.

**Rationale:** Character count produces inconsistent semantic density across languages and document types (a 1000-character chunk of dense legalese contains far fewer concepts than 1000 characters of sparse prose). Word count is a better proxy for semantic density and maps more intuitively to embedding model token limits.

**Consequences:**
- Embedding token limits are not guaranteed. A 1000-word chunk may still exceed 8191 tokens for very long words. In practice, 1000 words is well within `text-embedding-3-small`'s token limit.
- Character-count chunking is not supported by the ABCs — callers who need it must provide a custom `Chunker` implementation.

---

## ADR-004 — SectionChunker as default with WindowChunker fallback

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `SectionChunker` is the default chunker. When no Markdown or heading structure is detected in the document text, it falls back to `WindowChunker` behaviour automatically.

**Rationale:** Structure-aware chunking produces higher retrieval quality for structured documents (policies, reports, contracts) by keeping related content within the same section together. The fallback ensures the chunker never fails silently on unstructured documents like OCR output or raw text dumps.

**Consequences:**
- `section_title` is `None` for fallback window chunks — callers cannot assume it is always set.
- The fallback detection heuristic (heading marker regex) may misfire on documents with unusual formatting. Pass `chunker=WindowChunker(...)` explicitly to disable heuristic detection.

---

## ADR-005 — `openai` and `redis` as hard dependencies

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `openai` and `redis[hiredis]` are declared as core dependencies (not optional extras), even though they are only needed in production.

**Rationale:** Every production deployment needs both. Making them optional would require callers to remember to install extras, and version conflicts across optional and required dependencies have caused incidents in practice. The simpler install story (`pip install blue-data-doc-source`) is worth the extra disk space for dev installs.

**Consequences:**
- Development installs always pull in `openai` and `redis` even when only `DummyEmbeddingModel` and `MemoryVectorSource` are used.
- `DummyEmbeddingModel` and `MemoryVectorSource` are still the default in-process backends — importing `openai` does not mean it is called.

---

## ADR-006 — `EmbeddingModel.dim` property on the ABC

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `EmbeddingModel` requires a `dim` property. `DocumentPipeline` uses it to call `ensure_index(collection, dim=model.dim)` and validates every embedding against it on the first chunk.

**Rationale:** Dimension mismatches between the embedding model and the vector index are silent data corruption bugs — queries return garbage results rather than errors. Requiring `dim` at the type level makes mismatches fail immediately at ingest rather than at query time.

**Consequences:**
- All custom `EmbeddingModel` subclasses must implement `dim`. This is a minor friction for custom implementations.
- Changing the embedding model after initial indexing requires dropping and recreating the index (enforced by the dimension check on ingest).

---

## ADR-007 — `RecordStore` ABC with three implementations

**Date:** 2025-Q1  
**Status:** Accepted

**Decision:** Schema persistence (`DocumentRecord`) is abstracted behind a `RecordStore` ABC with three implementations: `MemoryRecordStore`, `FileRecordStore`, and `JsonSourceRecordStore`.

**Rationale:** Tests need a fast in-process store; development needs a store that survives restarts without Redis; production deployments with multiple API workers need a shared store backed by `RedisJSONSource`. A single ABC allows each deployment to pick the right backend without changing pipeline code.

**Consequences:**
- `MemoryRecordStore` is the default — callers who need persistence must explicitly pass `FileRecordStore` or `JsonSourceRecordStore`.
- `JsonSourceRecordStore` wraps any `JSONSource` from `blue-data`, making it composable with any `blue-data` backend.

---

## ADR-008 — GraphIndexer as an optional zero-overhead component

**Date:** 2025-Q1  
**Status:** Accepted

**Decision:** `graph_source=None` (the default) disables graph indexing entirely. `GraphIndexer` is a no-op when `graph_source` is not set — no graph client is created, no graph nodes or edges are written, no overhead is added to ingest.

**Rationale:** Most deployments do not need a knowledge graph. Requiring a Neo4j instance would significantly raise the setup bar and exclude the common single-document retrieval use case. The graph is an optional enrichment layer, not a requirement.

**Consequences:**
- `DOCUMENT.GRAPH_NEIGHBORS` and `DOCUMENT.GRAPH_QUERY` ops raise `DocSourceError` when `graph_source=None`.
- Entity nodes in the graph are shared across documents and are never removed by `DOCUMENT.DELETE` — callers must account for orphaned entity nodes when all documents mentioning an entity are deleted.

---

## ADR-009 — `to_dict()` omits `None` optional fields

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** `ChunkRecord.to_dict()` omits optional fields that are `None` rather than serialising them as JSON `null`.

**Rationale:** Redis Search tag and numeric fields cannot store `null` — presence of a field in the stored hash means it has a value. Storing `null` for optional fields like `page_start`, `section_title`, and `content_type` would pollute Redis hashes and require callers to distinguish `null` from "not present". Omitting absent fields keeps the Redis hash compact and avoids this ambiguity.

**Consequences:**
- Callers must use `record.get("page_start")` rather than `record["page_start"]`. Direct key access on optional fields will raise `KeyError` when the field is absent.
- The `embedding` field is also omitted by `to_dict()` — it is passed separately to `VectorSource.upsert()`.

---

## ADR-010 — PDF chunks are page-scoped (no cross-page chunks)

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** In V1, `PDFParser` produces one page dict per PDF page. Chunks are always scoped to a single page: `page_start == page_end`. Chunks never span page boundaries.

**Rationale:** Page provenance is critical for agent explainability — users need to know which page of a document a chunk came from. Allowing cross-page chunks would make provenance ambiguous. The cost is some loss of context at page boundaries, which is acceptable in V1.

**Consequences:**
- Short pages may produce very small chunks that fall below `chunk_size`. These are not merged with adjacent pages.
- Page-boundary context loss may affect retrieval quality for documents where key information spans page breaks. A future version may add optional page-boundary overlap.

---

## ADR-011 — `BasicEnricher` requires two or more title-case words for `PROPER_NOUN`

**Date:** 2025-Q1  
**Status:** Accepted

**Decision:** `BasicEnricher`'s PROPER_NOUN regex is `r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"` — it requires at least two consecutive title-case words.

**Rationale:** Single-word title-case matches (e.g. sentence-initial words like "The", "A", "In") produce massive false-positive rates. Requiring two words dramatically reduces noise while still catching entity names like "Acme Corp", "John Smith", and "United States".

**Consequences:**
- Single-word proper nouns (company codes, acronyms, single-name entities) are not detected by `BasicEnricher`. Use `LLMEnricher` when single-word entity detection matters.
- CamelCase names (e.g. "iPhone", "DeepMind") do not match the regex.

---

## ADR-012 — Platform layer contains no business logic

**Date:** 2024-Q4  
**Status:** Accepted

**Decision:** All three platform adapters (REST API, MCP server, CLI) delegate entirely to a `DocumentPipeline` singleton. No document processing logic lives in route handlers, MCP tool handlers, or CLI command functions.

**Rationale:** Business logic in the interface layer is duplicated across interfaces, untested in the library, and hard to evolve. The pipeline is the unit of correctness; the adapters are thin translation layers.

**Consequences:**
- The REST API, MCP server, and CLI are not independently testable without patching the `DocumentPipeline` singleton.
- Adapter tests patch the singleton and assert only on argument forwarding and output formatting — they do not test pipeline logic.

---

## ADR-013 — `NLQueryParser` uses Claude tool-use, not free-form parsing

**Date:** 2025-Q1  
**Status:** Accepted

**Decision:** `NLQueryParser` converts natural-language queries to `DocumentDataSource` op dicts using Claude's tool-use API rather than free-form completion parsing.

**Rationale:** Tool-use constrains the model's output to a well-typed schema. Free-form parsing requires a secondary parsing step that is fragile and hard to test. Tool-use eliminates the parsing step entirely and produces structured output directly.

**Consequences:**
- Requires Anthropic API access. `NLQueryParser` is not usable without an API key.
- Tool definitions must be kept in sync with the op code schema in `DocumentDataSource`. A new op requires both a code change and a tool definition update.
