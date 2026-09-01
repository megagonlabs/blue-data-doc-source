# blue-data-doc-source — API / Interface Spec

Three external interfaces sit above the core library: a Python library, a REST API, and a CLI. A fourth (MCP server) exists as a thin fastmcp adapter. No business logic belongs in any interface layer — all calls delegate to a `DocumentPipeline` singleton.

---

## 1. Python Library

The canonical interface. All other interfaces delegate to it.

### `DocumentPipeline`

The main orchestrator. Constructed once per process (see `platform/api/config.py` and `platform/cli/config.py` for singleton construction from env vars).

```python
class DocumentPipeline:
    def __init__(
        self,
        parser: Parser | None = None,           # default: auto-detect by extension
        chunker: Chunker | None = None,         # default: SectionChunker
        enricher: Enricher | None = None,       # default: BasicEnricher
        embedding_model: EmbeddingModel | None = None,  # default: DummyEmbeddingModel
        vector_source: VectorSource | None = None,
        graph_source: GraphSource | None = None,
        record_store: RecordStore | None = None,        # default: MemoryRecordStore
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None: ...
```

#### Methods

```python
def ensure_index(self, collection: str = "documents") -> None: ...
    # Create vector index for collection; idempotent

def ingest(self, path: str, collection: str = "documents") -> list[dict]: ...
    # Parse, chunk, enrich, embed, upsert; returns list[ChunkRecord.to_dict()]

def ingest_text(self, text: str, file_name: str, collection: str = "documents") -> list[dict]: ...
    # Ingest raw string; file_name used for mime_type detection

def ingest_pages(self, pages: list[dict], file_name: str, collection: str = "documents") -> list[dict]: ...
    # Ingest pre-parsed page dicts; skips Parser stage

def search(
    self,
    query: str,
    k: int = 5,
    collection: str = "documents",
    filter: dict | None = None,
    mode: str = "vector",
    rich: bool = False,
    context_chunks: int = 1,
) -> list[dict]: ...
    # KNN search; pre-filter reduces candidate set before KNN when filter provided
    # rich=True adds provenance block, neighbor chunks, by_document grouping

def find_records(self, filter: dict | None = None) -> list[dict]: ...
    # Structured filter over DocumentRecords; no embedding, no LLM

def find_then_search(
    self,
    query: str,
    find_filter: dict | None = None,
    k: int = 5,
    collection: str = "documents",
    mode: str = "vector",
) -> list[dict]: ...
    # Chain: find_records(find_filter) → extract document_ids → search with pre-filter

def get_record(self, document_id: str) -> dict | None: ...
    # Retrieve DocumentRecord by document_id

def delete(
    self,
    path: str | None = None,
    document_id: str | None = None,
    chunk_count: int | None = None,
    collection: str = "documents",
) -> int: ...
    # Remove all chunks for a document; returns count deleted

def graph_neighbors(
    self, entity_name: str, entity_type: str, k: int = 10
) -> list[dict]: ...
    # Requires graph_source; returns neighbor nodes via MENTIONS / CO_OCCURS_WITH

def graph_query(
    self, node_id: str, edge_type: str, direction: str = "out", k: int = 10
) -> list[dict]: ...
    # Requires graph_source; general graph traversal from any node ID
```

### `DocumentDataSource`

Extends blue-data's `DataSource` ABC. Public entry point for DataRegistry integration.

```python
class DocumentDataSource(DataSource):
    def execute_query(self, query: dict) -> Any: ...
```

#### Op codes

| Op | Required fields | Optional fields | Returns |
|----|----------------|----------------|---------|
| `DOCUMENT.ENSURE_INDEX` | — | `collection` | `None` |
| `DOCUMENT.INGEST` | `path` | `collection` | `list[dict]` (ChunkRecord dicts) |
| `DOCUMENT.SEARCH` | `query` | `k`, `collection`, `filter`, `mode`, `rich`, `context_chunks` | `list[dict]` |
| `DOCUMENT.FIND` | — | `filter` | `list[dict]` (DocumentRecord dicts) |
| `DOCUMENT.FIND_THEN_SEARCH` | `query` | `find_filter`, `k`, `collection`, `mode` | `list[dict]` |
| `DOCUMENT.GET_RECORD` | `document_id` | — | `dict \| None` |
| `DOCUMENT.DELETE` | at least one of: `path`, `document_id` | `chunk_count`, `collection` | `int` |
| `DOCUMENT.GRAPH_NEIGHBORS` | `entity_name`, `entity_type` | `k` | `list[dict]` |
| `DOCUMENT.GRAPH_QUERY` | `node_id`, `edge_type` | `direction`, `k` | `list[dict]` |

An unrecognised op raises `DocSourceError`.

---

## 2. REST API

Base URL: `http://localhost:8001`. All routes are under `/documents/`. Handlers are thin: all logic is in `DocumentPipeline`.

Request and response bodies are JSON. Request models are Pydantic `BaseModel`.

### `GET /documents/health`

Health check.

**Response** `200 OK`
```json
{"status": "ok"}
```

---

### `POST /documents/ensure-index`

Create the vector index for a collection (idempotent).

**Request**
```json
{"collection": "documents"}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `collection` | `string` | `"documents"` | Index name in the vector store |

**Response** `200 OK`
```json
{"status": "ok", "collection": "documents"}
```

---

### `POST /documents/ingest`

Ingest a document file from a local path.

**Request**
```json
{"path": "/data/report.pdf", "collection": "documents"}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `path` | `string` | required | Absolute path to the file to ingest |
| `collection` | `string` | `"documents"` | Target vector index collection |

**Response** `200 OK`
```json
{
    "document_id": "a3f2...b7e1",
    "file_name": "report.pdf",
    "chunk_count": 42,
    "collection": "documents"
}
```

---

### `POST /documents/search`

Semantic or keyword search over ingested chunks.

**Request**
```json
{
    "query": "eligibility requirements for coverage",
    "k": 5,
    "collection": "documents",
    "filter": {"mime_type": "application/pdf"},
    "mode": "vector",
    "rich": false,
    "context_chunks": 1
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | `string` | required | Natural-language query |
| `k` | `integer` | `5` | Number of results |
| `collection` | `string` | `"documents"` | Collection to search |
| `filter` | `object \| null` | `null` | Pre-filter applied before KNN |
| `mode` | `string` | `"vector"` | `"vector"` or `"hybrid"` |
| `rich` | `boolean` | `false` | Add provenance + neighbor context + by_document grouping |
| `context_chunks` | `integer` | `1` | Neighbor chunks to add on each side when `rich=true` |

**Response** `200 OK` — array of ChunkRecord dicts (or rich result dicts when `rich=true`).

---

### `POST /documents/find`

Structured filter over DocumentRecords — no embedding, no LLM.

**Request**
```json
{"filter": {"document_type": "contract", "fields.governing_law": "California"}}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `filter` | `object \| null` | `null` | Dot-notation field paths; operators: `contains`, `lt`, `gt`, `lte`, `gte` |

**Response** `200 OK` — array of DocumentRecord dicts.

```json
[
    {
        "document_id": "a3f2...b7e1",
        "document_type": "contract",
        "confidence": 0.92,
        "classifier": "keyword",
        "fields": {"parties": ["Acme Corp", "Globex"], "governing_law": "California"}
    }
]
```

---

### `POST /documents/find-then-search`

Chain: filter DocumentRecords → restrict search to matched document IDs.

**Request**
```json
{
    "query": "termination clause",
    "find_filter": {"document_type": "contract"},
    "k": 5,
    "collection": "documents",
    "mode": "vector"
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | `string` | required | Search query |
| `find_filter` | `object \| null` | `null` | Filter passed to `find_records()` |
| `k` | `integer` | `5` | Number of results |
| `collection` | `string` | `"documents"` | Collection to search |
| `mode` | `string` | `"vector"` | `"vector"` or `"hybrid"` |

**Response** `200 OK` — array of ChunkRecord dicts.

---

### `GET /documents/record/{document_id}`

Retrieve the DocumentRecord for a specific document.

**Path parameter:** `document_id` — SHA-256 hex string.

**Response** `200 OK`
```json
{
    "document_id": "a3f2...b7e1",
    "document_type": "invoice",
    "confidence": 0.88,
    "classifier": "keyword",
    "fields": {"vendor": "Acme Corp", "total_due": "$1,250.00"}
}
```

**Response** `404 Not Found` when `document_id` is not in the record store.

---

### `DELETE /documents/document`

Delete all chunks for a document from the vector store.

**Request**
```json
{"path": "/data/report.pdf", "collection": "documents"}
```

Provide `path` **or** (`document_id` + `chunk_count`):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `path` | `string \| null` | `null` | File path; pipeline re-derives `document_id` from SHA-256 |
| `document_id` | `string \| null` | `null` | Direct document ID |
| `chunk_count` | `integer \| null` | `null` | Used when `document_id` provided without path |
| `collection` | `string \| null` | `null` | Defaults to configured collection |

**Response** `200 OK`
```json
{"deleted": 42}
```

---

## 3. MCP Server

Entry point: `platform/document-mcp/src/server.py`. Transport: **stdio** (fastmcp).

All tools return a JSON-encoded string. The pipeline singleton is constructed once from the same env vars as the REST API.

### `ensure_index(collection="documents")`

Create the vector index. Idempotent.

**Returns** JSON string:
```json
{"status": "ok", "collection": "documents"}
```

---

### `ingest_document(path, collection="documents")`

Ingest a file from a local path.

**Returns** JSON string:
```json
{"document_id": "a3f2...b7e1", "file_name": "report.pdf", "chunk_count": 42}
```

---

### `search_documents(query, collection="documents", k=5)`

Semantic search over ingested chunks.

**Returns** JSON string:
```json
{
    "results": [
        {"chunk_id": "a3f2..._chunk_0003", "text": "...", "score": 0.91, ...}
    ]
}
```

---

### `delete_document(path=None, document_id=None, chunk_count=None, collection="documents")`

Remove all chunks for a document. Provide `path` or `document_id`.

**Returns** JSON string:
```json
{"deleted": 42}
```

---

## 4. CLI

Package: `blue-data-doc-cli`. Entry point: `blue-data-doc`.

Global option: `--json` — print output as JSON instead of plain text.

```
blue-data-doc [--json] <command> [options]
```

### `ensure-index`

```
blue-data-doc ensure-index [--collection COLLECTION]
```

Creates the vector index. Prints `"Index ready: <collection>"` on success.

### `ingest`

```
blue-data-doc ingest PATH [--collection COLLECTION]
```

Ingests the file at `PATH`. Prints `"Ingested <file_name>: <chunk_count> chunks (<document_id>)"` on success.

### `search`

```
blue-data-doc search QUERY [--collection COLLECTION] [-k N]
```

Searches for `QUERY`. Prints each result as `[score] chunk_id\n  text preview`. With `--json`, prints the full result array.

### `delete`

```
blue-data-doc delete [--path FILE_PATH | --id DOCUMENT_ID --chunks CHUNK_COUNT] [--collection COLLECTION]
```

Deletes all chunks for the document. Provide `--path` or `--id + --chunks`. Prints `"Deleted <n> chunks"`.

---

## Environment Variables

All three platforms (API, MCP, CLI) read the same env vars to construct the `DocumentPipeline` singleton.

| Variable | Default | Notes |
|----------|---------|-------|
| `BLUE_DOC_EMBEDDING` | `dummy` | `openai` or `dummy` |
| `OPENAI_API_KEY` | — | Required when `BLUE_DOC_EMBEDDING=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Ignored when `BLUE_DOC_EMBEDDING=dummy` |
| `BLUE_DOC_REDIS_HOST` | — | Omit to run without vector store (MemoryVectorSource used) |
| `BLUE_DOC_REDIS_PORT` | `6379` | Ignored when `BLUE_DOC_REDIS_HOST` is unset |
| `BLUE_DOC_COLLECTION` | `documents` | Default collection name |
| `BLUE_DOC_CHUNK_SIZE` | `1000` | Word-count target per chunk |
| `BLUE_DOC_OVERLAP` | `150` | Word-count overlap between consecutive window chunks |
| `BLUE_DOC_OCR_FALLBACK` | — | Set to `1` or `true` to enable Tesseract OCR for scanned PDFs |
