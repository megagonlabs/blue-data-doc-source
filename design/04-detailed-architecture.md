# blue-data-doc-source — Detailed Architecture

## External Interfaces

All four interfaces are thin adapters over the Python library. Each delegates to `DocumentDataSource.execute_query` or `DocumentPipeline` directly.

| Interface | Technology | Primary Use |
|-----------|-----------|-------------|
| Python Library | Native Python API | Embedding document ingestion in Python applications and pipelines |
| MCP Server | fastmcp, stdio | LLM agent tool use (Claude Desktop, Claude Code, etc.) |
| REST API | FastAPI, port 8001 | Non-Python consumers and microservice deployments |
| CLI | Click (`blue-data-doc`) | Development, batch ingestion, and scripting |

---

## Information Flow

### Ingest Path (`DOCUMENT.INGEST`)

1. Caller provides a file path
2. **`document_id`** is computed: `sha256(file_bytes).hexdigest()` — the canonical identity of the document, independent of its path or name
3. **Parser** reads the file and returns `list[{"text": str, "page_num"?: int}]`; returns `[]` on unreadable or empty documents
4. **DocumentClassifier** scores the first 500 words against signal phrase sets → `(document_type, confidence)` (zero API cost with `KeywordClassifier`)
5. **FieldExtractor** calls the LLM once per document to extract type-specific fields (parties, dates, totals, etc.) → `fields: dict`
6. **RecordStore** persists the `DocumentRecord` (`{document_id, document_type, confidence, classifier, fields}`) keyed by `document_id`
7. **Chunker** splits each page's text into `(section_title, text)` pairs; `SectionChunker` detects headings and groups content by section; falls back to window chunking when no headings are found; chunks are never empty
8. **Enricher** annotates each chunk: `{"entities": [...], "semantic_hints": {...}}`
9. **EmbeddingModel** converts each chunk text to a float vector; length validated against `model.dim` on the first chunk
10. **ChunkRecord** is assembled for each chunk: `chunk_id = {document_id}_chunk_{index:04d}`, plus all provenance fields (`page_start`, `page_end`, `section_title`, `mime_type`, `document_type`, `entities`, `semantic_hints`, `metadata`)
11. `ChunkRecord.to_dict()` omits `None` optional fields; the embedding is passed separately
12. **VectorSource** upserts each chunk — same `chunk_id` → upsert overwrites; re-ingest is idempotent
13. **GraphIndexer** (if `graph_source` is set): creates `Document`, `Section`, `Chunk`, and `Entity` nodes and wires edges; skipped entirely when `graph_source=None`
14. Returns `list[ChunkRecord.to_dict()]`

PDF chunks are page-scoped: `page_start == page_end`. Chunks never span page boundaries.

---

### Search Path (`DOCUMENT.SEARCH`)

1. Caller provides a query string and optional `k`, `collection`, `filter`, `mode`, `rich`, `context_chunks`
2. **EmbeddingModel** embeds the query text → float vector
3. If `filter` is provided: a Redis Search pre-filter expression is built from the filter dict and passed to `VectorSource` to restrict the candidate set before KNN
4. **VectorSource.KNN** returns the `k` nearest chunks
5. If `rich=True`: for each result, the pipeline fetches the `context_chunks` predecessor and successor chunks (via `NEXT`/`PREV` edges in the graph or by `chunk_index` arithmetic); assembles a provenance block (page, section, document_type); groups results by document
6. Returns a list of ChunkRecord dicts (or rich result dicts when `rich=True`)

---

### Ensure Index Path (`DOCUMENT.ENSURE_INDEX`)

1. Caller provides a collection name
2. **DocumentPipeline** calls `VectorSource.create_index(collection, dim=model.dim)` with six extra TAG/NUMERIC fields: `document_id`, `file_name`, `section_title`, `mime_type`, `page_start`, `content_type`
3. Idempotent — skips creation if the index already exists

Must be called before the first ingest when using a vector store. `DocumentPipeline` does not auto-create the index.

---

### Find Path (`DOCUMENT.FIND`)

1. Caller provides a filter dict
2. **RecordStore** iterates all `DocumentRecord` entries and applies the filter
3. Filter supports dot-notation field access and operators: `contains`, `lt`, `gt`, `lte`, `gte`; currency symbols and commas stripped before numeric comparison
4. Returns matching `DocumentRecord` dicts — no embedding, no LLM, no vector store access

---

### Find-then-Search Path (`DOCUMENT.FIND_THEN_SEARCH`)

1. `DOCUMENT.FIND` runs with `find_filter` → returns matching `DocumentRecord` dicts
2. `document_id` values are extracted from the matched records
3. `DOCUMENT.SEARCH` runs with a pre-filter restricted to those `document_id` values
4. Returns ranked chunk dicts from within the matched document set

---

### Get Record Path (`DOCUMENT.GET_RECORD`)

1. Caller provides a `document_id`
2. **RecordStore.get(`document_id`)** returns the stored `DocumentRecord` dict, or `None` if not found

---

### Delete Path (`DOCUMENT.DELETE`)

1. Caller provides `path` or (`document_id` + `chunk_count`)
2. If `path` is given: `document_id` is derived from `sha256(file_bytes)`; `chunk_count` is determined from the stored record or by scanning the vector index
3. **VectorSource** deletes all chunk vectors for this `document_id`
4. `Document`, `Section`, and `Chunk` graph nodes are removed (if `graph_source` is set); `Entity` nodes are **not** removed — they are shared across documents
5. Returns the count of deleted chunks

---

### Graph Neighbors Path (`DOCUMENT.GRAPH_NEIGHBORS`)

1. Caller provides `entity_name`, `entity_type`, and optional `k`
2. **GraphIndexer** looks up the entity node `entity:{entity_type}:{name.lower()}`
3. Returns `k` neighbor nodes reachable via `MENTIONS`, `CO_OCCURS_WITH`, or `APPEARS_IN` edges

Requires `graph_source` to be set — raises `DocSourceError` otherwise.

---

### Graph Query Path (`DOCUMENT.GRAPH_QUERY`)

1. Caller provides `node_id`, `edge_type`, `direction` (`"in"` or `"out"`), and optional `k`
2. **GraphSource** traverses from `node_id` along edges of `edge_type` in the given direction
3. Returns the `k` reachable neighbor nodes as dicts

General-purpose traversal — works from any node ID (Document, Section, Chunk, or Entity).

---

## Component Contracts

### `Parser`

```python
class Parser(ABC):
    def parse(self, path: str) -> list[dict]: ...
    # Returns [{"text": str, "page_num"?: int}]; [] on unreadable/empty

    def parse_blocks(self, path: str) -> list[dict]: ...
    # Returns typed blocks: {"type": "text"|"table", "text": str, "table"?: {...}, "page_num"?: int}
```

`parse()` must always return a list. Unreadable files return `[]` — the pipeline returns `[]` chunks in that case.

---

### `Chunker`

```python
class Chunker(ABC):
    def chunk_with_titles(self, text: str) -> list[tuple[str | None, str]]: ...
    # Returns [(section_title, chunk_text)]
    # section_title is None for window/sentence chunkers and for pre-heading content
```

Chunk text is never empty — `DocumentPipeline` skips chunks where `text.strip()` is falsy.

---

### `Enricher`

```python
class Enricher(ABC):
    def enrich(self, text: str) -> dict: ...
    # Returns {"entities": [{"text": str, "type": str}], "semantic_hints": dict}
    # Always returns both keys; empty lists/dicts are valid
```

---

### `EmbeddingModel`

```python
class EmbeddingModel(ABC):
    @property
    def dim(self) -> int: ...      # used by ensure_index; validated on first chunk

    def embed_text(self, text: str) -> list[float]: ...
    # len(result) must equal self.dim
```

---

### `ChunkRecord`

```python
@dataclass
class ChunkRecord:
    # Required — always in to_dict()
    document_id:    str        # sha256(file_bytes).hexdigest()
    chunk_id:       str        # {document_id}_chunk_{index:04d}
    text:           str
    file_name:      str
    source_type:    str        # "document"
    mime_type:      str        # "text/plain" | "application/pdf" | etc.
    entities:       list[dict]
    semantic_hints: dict
    metadata:       dict       # ingested_at, etc.

    # Optional — omitted from to_dict() when None
    page_start:     int | None   # 1-based; set by PDF parsers; page_start == page_end
    page_end:       int | None
    section_title:  str | None
    document_type:  str | None
    content_type:   str | None   # "text" | "table"
    table:          dict | None  # {"headers": [...], "rows": [...]}
    embedding:      list[float] | None  # never in to_dict(); passed separately to VectorSource
```

---

### Graph node and edge schema

```
(Document {document_id, file_name, document_type, mime_type, ingested_at, chunk_count})
    -[:HAS_SECTION]->
(Section {section_id, title, document_id, position})
    -[:HAS_CHUNK]->
(Chunk {chunk_id, document_id, text_preview, chunk_index, page?, section_title?})
    -[:MENTIONS]->
(Entity {name, entity_type})

Additional edges:
  Document  -[:HAS_CHUNK]->       Chunk    (direct, regardless of section)
  Chunk     -[:NEXT]->            Chunk    (sequential forward)
  Chunk     -[:PREV]->            Chunk    (sequential backward)
  Entity    -[:CO_OCCURS_WITH]->  Entity   (both entities in same chunk)
  Entity    -[:APPEARS_IN]->      Document (entity present anywhere in document)
```

Node IDs are deterministic: `doc:{document_id}`, `section:{document_id}:{title.lower()}`, `chunk:{chunk_id}`, `entity:{entity_type}:{name.lower()}`.

---

## Dependency Graph

```
blue-data-doc-source (core library)
    ├── blue-data              DataSource / VectorSource / GraphSource ABCs + backends
    ├── openai ≥1.0.0          OpenAIEmbeddingModel; LLMFieldExtractor; LLMEnricher; StructuredFieldExtractor; NLQueryParser
    ├── pypdf ≥4.0.0           PDFParser (text extraction from digital PDFs)
    ├── python-docx ≥1.0.0     DocxParser (heading-preserving DOCX extraction)
    └── redis[hiredis] ≥4.5.0  RedisVectorSource + RedisJSONSource backends

Optional extras:
    pytesseract + pdf2image  [ocr]   — OCR fallback for scanned PDFs
    nltk                     [nlp]   — SentenceChunker
    mcp                      [mcp]   — fastmcp runtime for MCP server

Platform services:
    platform/doc-api/            FastAPI + Uvicorn (REST API, port 8001)
    platform/doc-mcp/   fastmcp (MCP server, stdio)
    platform/doc-cli/            Click (CLI, blue-data-doc command)
```

---

## Pipeline Stage Ownership

Each subdirectory of `lib/src/blue_data_doc_source/source/document/` owns exactly one pipeline stage:

```
parsers/      →  schema/     →  chunkers/   →  enrichment/  →  embeddings/  →  chunk_record.py  →  graph/
   │                │                │               │               │                │               │
Parser ABC    Classifier +      Chunker ABC     Enricher ABC   EmbeddingModel   ChunkRecord      GraphIndexer
TextParser    FieldExtractor    SectionChunker  BasicEnricher  OpenAIEmbedding  dataclass
PDFParser     KeywordClass.     WindowChunker   LLMEnricher    DummyEmbedding
PlumberPDF    LLMExtractor      SentenceChunker
DocxParser    RecordStore
```

`pipeline.py` orchestrates the chain. `document_source.py` sits above it, dispatching on op codes. Neither contains business logic — they coordinate stage calls.
