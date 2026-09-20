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
doc ensure-index

# Ingest a document
doc ingest reports/policy.pdf

# Search
doc search "termination clause" -k 5

# Delete by file path (same session)
doc delete --path reports/policy.pdf

# Delete by document_id + chunk_count (cross-session)
doc delete --id <document_id> --chunks <chunk_count>

# Output raw JSON for any command
doc --json search "termination clause"
```

## Quick start (no API key, no Redis)

```bash
doc ingest data/sample_policy.txt
# chunks printed to stdout; no vector store required
```

## Quick start (OpenAI + Redis)

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest

export BLUE_DOC_EMBEDDING=openai
export OPENAI_API_KEY=sk-...
export BLUE_DOC_REDIS_HOST=localhost

doc ensure-index
doc ingest contract.pdf
doc search "governing law"
```
