# CLI

`blue-data-doc-cli` wraps the full pipeline as a terminal command. Supports `.txt`, `.pdf`, and `.docx`.

> Full platform documentation (API, CLI, MCP) is in [platform/README.md](../platform/README.md). This page is the user-facing quick reference.

## Install

```bash
pip install -e lib/ -e platform/doc-cli/
```

## Configuration

All configuration is via environment variables:

| Variable | Description | Default |
|---|---|---|
| `BLUE_DOC_EMBEDDING` | `openai` or `dummy` | `dummy` |
| `OPENAI_API_KEY` | Required when `BLUE_DOC_EMBEDDING=openai` | — |
| `BLUE_DOC_REDIS_HOST` | Redis host; omit to run without a vector store | — |
| `BLUE_DOC_REDIS_PORT` | Redis port | `6379` |
| `BLUE_DOC_COLLECTION` | Default collection name | `documents` |
| `BLUE_DOC_CHUNK_SIZE` | Max words per chunk | `1000` |
| `BLUE_DOC_OVERLAP` | Word overlap between chunks | `150` |
| `BLUE_DOC_OCR_FALLBACK` | `1` or `true` to enable Tesseract OCR on scanned PDFs | off |

## Commands

```bash
# Create the vector index before first ingest (idempotent)
blue-docdb doc ensure-index

# Ingest a document
blue-docdb doc ingest reports/policy.pdf

# Search
blue-docdb doc search "termination clause" -k 5

# Delete by file path (same session)
blue-docdb doc delete --path reports/policy.pdf

# Delete by document_id + chunk_count (cross-session)
blue-docdb doc delete --id <document_id> --chunks <chunk_count>

# Output as JSON
blue-docdb --format json doc search "termination clause"

# Output as newline-delimited JSON (for piping)
blue-docdb --format nl doc search "termination clause" | jq .chunk_id
```

## Output formats

| Format | Description |
|---|---|
| `table` | Human-readable, aligned columns (default) |
| `json` | Pretty-printed JSON |
| `nl` | Newline-delimited JSON — one object per line, suitable for `jq`, `grep`, shell scripts |

## Error codes

| Code | Meaning |
|---|---|
| `NOT_FOUND` | File or resource does not exist |
| `INVALID_ARGS` | Missing or incompatible arguments |
| `INGEST_FAILED` | Parsing, chunking, or embedding error during ingest |
| `SEARCH_FAILED` | Query execution error |
| `DELETE_FAILED` | Chunk removal error |
| `CONFIG_ERROR` | Pipeline configuration or index creation error |

Errors are written to stderr. In `--format json` mode, errors are structured:

```json
{"error": {"code": "NOT_FOUND", "message": "File not found: foo.txt"}}
```

## Quick start (no API key, no Redis)

```bash
blue-docdb doc ingest data/sample_policy.txt
# chunks printed to stdout; no vector store required
```

## Quick start (OpenAI + Redis)

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest

export BLUE_DOC_EMBEDDING=openai
export OPENAI_API_KEY=sk-...
export BLUE_DOC_REDIS_HOST=localhost

blue-docdb doc ensure-index
blue-docdb doc ingest contract.pdf
blue-docdb doc search "governing law"
```
