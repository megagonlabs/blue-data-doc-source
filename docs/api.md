# REST API

`blue-data-doc-api` is a FastAPI server that exposes the full document pipeline over HTTP. Interactive docs are available at `http://localhost:8001/docs` when the server is running.

> Full platform documentation (API, CLI, MCP) is in [platform/README.md](../platform/README.md). This page is the user-facing quick reference.

## Install and run

```bash
pip install -e lib/ -e platform/api/

# Dev server with auto-reload
uvicorn doc_api.main:app --reload --port 8001

# Or via the installed script
blue-data-doc-api
```

## Configuration

Same environment variables as the CLI — see [cli.md](cli.md#configuration).

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

```bash
# Ensure index
curl -X POST http://localhost:8001/documents/ensure-index \
     -H "Content-Type: application/json" -d '{}'

# Ingest
curl -X POST http://localhost:8001/documents/ingest \
     -H "Content-Type: application/json" \
     -d '{"path": "/data/contract.pdf"}'

# Search
curl -X POST http://localhost:8001/documents/search \
     -H "Content-Type: application/json" \
     -d '{"query": "governing law", "k": 5}'

# Rich search with provenance
curl -X POST http://localhost:8001/documents/search \
     -H "Content-Type: application/json" \
     -d '{"query": "termination clause", "rich": true}'

# Find by extracted field
curl -X POST http://localhost:8001/documents/find \
     -H "Content-Type: application/json" \
     -d '{"filter": {"document_type": "contract", "fields.total_due": {"gt": 30000}}}'

# Delete by document_id
curl -X DELETE http://localhost:8001/documents/document \
     -H "Content-Type: application/json" \
     -d '{"document_id": "doc_3a7f1c2e", "chunk_count": 12}'
```
