# API Reference

Request and response schemas for the `blue-data-doc-api` REST server. All endpoints are under the `/documents` prefix.

See [api.md](api.md) for setup and example requests.

---

## `GET /documents/health`

No request body.

**Response**
```json
{"status": "ok"}
```

---

## `POST /documents/ensure-index`

Create the vector index for a collection. Idempotent — safe to call multiple times.

**Request**
```json
{"collection": "contracts"}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `collection` | string | `BLUE_DOC_COLLECTION` env var | Collection to index |

**Response**
```json
{"status": "ok", "collection": "contracts"}
```

---

## `POST /documents/ingest`

Parse, chunk, enrich, embed, and upsert a document. Re-ingesting the same file is safe — content-hash deduplication means the same file always produces the same chunk IDs.

**Request**
```json
{"path": "/data/contract.pdf", "collection": "contracts"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | ✅ | Absolute path to the file on the server filesystem |
| `collection` | string | — | Overrides `BLUE_DOC_COLLECTION` |

**Response**
```json
{
  "document_id": "bc66f8ef...",
  "file_name": "contract.pdf",
  "chunk_count": 12,
  "collection": "contracts"
}
```

**Errors**: `404` file not found · `400` unsupported format · `422` empty file

---

## `POST /documents/search`

Embed a query and return the k most relevant chunks.

**Request**
```json
{
  "query": "termination clause",
  "k": 5,
  "collection": "contracts",
  "filter": {"document_id": "bc66f8ef..."},
  "mode": "vector",
  "rich": false,
  "context_chunks": 1
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | ✅ | Query text |
| `k` | int | `5` | Number of results |
| `collection` | string | env default | Collection to search |
| `filter` | object | — | Pre-filter applied before KNN (e.g. `{"document_id": "..."}`) |
| `mode` | string | `"vector"` | Search mode: `"vector"` (KNN cosine), `"bm25"` (keyword, no embeddings needed), `"hybrid"` (RRF fusion of KNN + BM25), `"rerank"` (BM25 shortlist re-ranked by cosine). **Note:** `"bm25"`, `"hybrid"`, and `"rerank"` require documents to have been ingested in the current server process — they use an in-memory index that is lost on restart. |
| `rich` | bool | `false` | Return structured response with provenance + `by_document` grouping |
| `context_chunks` | int | `1` | Neighbor chunks to include per result when `rich=true` |

**Response (rich=false)** — list of chunk dicts with `chunk_id`, `text`, `score`, `file_name`, `section_title`, `page_start`, `page_end`.

**Response (rich=true)** — structured object with `results` list (each entry has `chunk`, `neighbors`, `citation`) and `by_document` grouping.

---

## `POST /documents/find`

Filter `DocumentRecord`s by extracted fields. No embedding, no LLM — deterministic.

**Request**
```json
{"filter": {"document_type": "contract", "fields.total_due": {"gt": 30000}}}
```

Filter operators: `gt`, `gte`, `lt`, `lte`, `eq`, `contains`. All conditions are ANDed. Pass `{}` to return all records.

**Response** — list of `DocumentRecord` dicts with `document_id`, `file_name`, `document_type`, `confidence`, `fields`.

---

## `POST /documents/find-then-search`

Filter documents by fields, then run vector search within the matched set.

**Request**
```json
{
  "query": "indemnification",
  "find_filter": {"document_type": "contract"},
  "k": 5,
  "collection": "contracts",
  "mode": "vector"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | ✅ | Query text |
| `find_filter` | object | — | Field filter — same syntax as `/find` |
| `k` | int | `5` | Number of results |
| `collection` | string | env default | Collection to search |
| `mode` | string | `"vector"` | Search mode: `"vector"`, `"bm25"`, `"hybrid"`, or `"rerank"` |

**Response** — ranked list of chunk dicts, same shape as `/search`.

---

## `GET /documents/record/{document_id}`

Retrieve the `DocumentRecord` written at ingest time.

**Response**
```json
{
  "document_id": "bc66f8ef...",
  "file_name": "contract.pdf",
  "document_type": "contract",
  "confidence": 0.85,
  "fields": {"parties": ["Acme Corp", "Blue Shield"], "governing_law": "California"}
}
```

**Errors**: `404` no record for that `document_id`

---

## `DELETE /documents/document`

Delete all chunks for a document from the vector store.

**Request** — provide `path` or `document_id` + `chunk_count`:

```json
{"document_id": "bc66f8ef...", "chunk_count": 12, "collection": "contracts"}
```

| Field | Type | Description |
|---|---|---|
| `path` | string | Path to source file (same server session) |
| `document_id` | string | From a previous ingest response (cross-session) |
| `chunk_count` | int | Required when using `document_id` |
| `collection` | string | Overrides `BLUE_DOC_COLLECTION` |

**Response**
```json
{"deleted": 12}
```
