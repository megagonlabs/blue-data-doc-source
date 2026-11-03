# REST API

`blue-data-doc-api` is a FastAPI server that exposes the full document pipeline over HTTP. Interactive docs are available at `http://localhost:8001/docs` when the server is running.

> Full platform documentation (API, CLI, MCP) is in [platform/README.md](../platform/README.md). This page is the user-facing quick reference.

## Install and run

```bash
pip install -e lib/ -e platform/doc-api/

# Set an API key — every data route requires it (see Authentication below)
export BLUE_DOC_API_KEY=choose-a-secret

# Dev server with auto-reload
uvicorn doc_api.main:app --reload --port 8001

# Or via the installed script
blue-data-doc-api          # binds 127.0.0.1:8001 by default
```

## Configuration

Pipeline settings share the CLI's environment variables — see
[cli.md](cli.md#configuration). The server adds the shared `blue_data.server`
policy (loopback bind + mandatory API key):

| Variable | Default | Effect |
|---|---|---|
| `BLUE_DOC_API_HOST` | `127.0.0.1` | Bind host — loopback only unless overridden |
| `BLUE_DOC_API_PORT` | `8001` | Bind port (family port map: registry 8000, doc 8001, web 8002) |
| `BLUE_DOC_API_KEY` | _(unset)_ | Required on every data route; send as `X-API-Key: <key>` or `Authorization: Bearer <key>`. With no key configured, data routes return **401** |
| `BLUE_DOC_API_AUTH_DISABLED` | _(unset)_ | Set truthy (`1`/`true`/`yes`) to run with no API key (local/dev only) |
| `BLUE_DOC_ROOT` | _(unset)_ | When set, `/documents/ingest` and `/documents/delete` accept only a **filename resolved inside this directory** (no separators, absolute paths, or `..`); anything else returns **400** |

## Authentication

Every route except `GET /documents/health` and the docs routes (`/docs`,
`/redoc`, `/openapi.json`) requires the API key. Send it on every request:

```bash
curl -X POST http://localhost:8001/documents/search \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "governing law", "k": 5}'
```

- **`401 Unauthorized`** — API key missing or wrong (or none configured on the server).
- **`415 Unsupported Media Type`** — state-changing routes (`POST`/`DELETE`) require an `application/json` body; form, multipart, or text bodies are refused (CSRF protection).

> The default bind is loopback. Set `BLUE_DOC_API_HOST=0.0.0.0` only behind a
> trusted proxy or network boundary, and always with `BLUE_DOC_API_KEY` set.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents/health` | Health check |
| `POST` | `/documents/ensure-index` | Create vector index — call once before first ingest |
| `POST` | `/documents/ingest` | Parse, chunk, embed, and upsert a document |
| `POST` | `/documents/search` | Semantic search; add `"rich": true` for provenance + grouping |
| `POST` | `/documents/find` | Filter DocumentRecords by extracted fields (no embedding) |
| `POST` | `/documents/find-then-search` | Filter by fields, then search within matched documents |
| `GET` | `/documents/record/{document_id}` | Retrieve the schema record written at ingest |
| `DELETE` | `/documents/document` | Delete all chunks for a document |

See [api-reference.md](api-reference.md) for full request/response schemas.

## Example requests

Every request sends `-H "X-API-Key: $BLUE_DOC_API_KEY"` (export the key first,
as in [Install and run](#install-and-run)).

```bash
# Ensure index
curl -X POST http://localhost:8001/documents/ensure-index \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" -d '{}'

# Ingest (absolute path — allowed only when BLUE_DOC_ROOT is unset)
curl -X POST http://localhost:8001/documents/ingest \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"path": "/data/contract.pdf"}'

# Ingest when BLUE_DOC_ROOT is set — pass just the filename inside that root
curl -X POST http://localhost:8001/documents/ingest \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"path": "contract.pdf"}'

# Search
curl -X POST http://localhost:8001/documents/search \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "governing law", "k": 5}'

# Rich search with provenance
curl -X POST http://localhost:8001/documents/search \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "termination clause", "rich": true}'

# Find by extracted field
curl -X POST http://localhost:8001/documents/find \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"filter": {"document_type": "contract", "fields.total_due": {"gt": 30000}}}'

# Delete by document_id
curl -X DELETE http://localhost:8001/documents/document \
     -H "X-API-Key: $BLUE_DOC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"document_id": "doc_3a7f1c2e", "chunk_count": 12}'
```
