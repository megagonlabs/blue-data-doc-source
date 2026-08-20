# Platform

The `platform/` directory contains deployable components built on top of `lib/`. Install `lib` first — all components depend on it.

| Component | Description |
|---|---|
| `cli` | Command-line interface — ingest, search, and delete documents from a terminal |
| `document-mcp` | MCP server — exposes document operations as AI-callable tools for Claude and other MCP clients |

## Installation

```bash
# Install the library first
pip install -e lib/

# CLI
pip install -e platform/cli/

# MCP server
pip install -e platform/document-mcp/
```

---

## CLI (`platform/cli`)

The `doc` command wraps the full pipeline for terminal use. Supports `.txt`, `.pdf`, and `.docx`.

### Configuration

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

### Commands

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
doc delete --id doc_3a7f1c2e --chunks 12

# Output raw JSON for any command
doc --json search "termination clause"
```

### Quick start (no API key, no Redis)

```bash
pip install -e lib/ -e platform/cli/
doc ingest mydoc.txt          # chunks printed to stdout, no vector store
doc search "my query"         # requires BLUE_DOC_REDIS_HOST
```

### Quick start (OpenAI + Redis)

```bash
docker run -p 6379:6379 redis/redis-stack-server:latest

export BLUE_DOC_EMBEDDING=openai
export OPENAI_API_KEY=sk-...
export BLUE_DOC_REDIS_HOST=localhost

doc ensure-index
doc ingest contract.pdf
doc search "governing law"
```

---

## MCP Server (`platform/document-mcp`)

Exposes document operations as MCP tools so Claude and other MCP clients can ingest, search, and delete documents directly.

### Tools

| Tool | Description |
|---|---|
| `ensure_index` | Create the vector index for a collection — call once before first ingest |
| `ingest_document` | Parse, chunk, embed, and upsert a file; returns `document_id` and `chunk_count` |
| `search_documents` | Embed a query and return the k most relevant chunks |
| `delete_document` | Remove all chunks for a document by path or `document_id` + `chunk_count` |

### Configuration

Same environment variables as the CLI (see table above).

### Running

```bash
# In-memory, dummy embeddings (no deps beyond lib)
BLUE_DOC_EMBEDDING=dummy python platform/document-mcp/src/server.py

# With Redis and OpenAI embeddings
BLUE_DOC_EMBEDDING=openai \
OPENAI_API_KEY=sk-... \
BLUE_DOC_REDIS_HOST=localhost \
python platform/document-mcp/src/server.py
```

### Claude Desktop / Claude Code integration

Add to your MCP config:

```json
{
  "mcpServers": {
    "blue-doc-source": {
      "command": "python",
      "args": ["platform/document-mcp/src/server.py"],
      "env": {
        "BLUE_DOC_EMBEDDING": "openai",
        "OPENAI_API_KEY": "sk-...",
        "BLUE_DOC_REDIS_HOST": "localhost"
      }
    }
  }
}
```

See `examples/framework-demos/mcp_demo.py` for a full runnable demo.
