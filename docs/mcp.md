# MCP Server

The MCP server exposes blue-data-doc-source as four AI-callable tools over the Model Context Protocol. Any MCP-compatible agent — Claude Desktop, Claude Code, or a custom agent using the MCP SDK — can ingest documents, search them, and manage the vector index without any custom integration code.

## Tools

| Tool | Description |
|---|---|
| `ingest_document` | Parse, chunk, enrich, embed, and upsert a document |
| `search_documents` | Semantic search over ingested documents |
| `ensure_index` | Create the vector index before first ingest |
| `delete_document` | Remove all chunks for a document from the vector store |

## Installation

```bash
pip install -e ".[mcp]"
```

OCR support (optional):

```bash
pip install -e ".[mcp,ocr]"
brew install tesseract poppler   # macOS
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `BLUE_DOC_EMBEDDING` | `dummy` | `openai` or `dummy` |
| `OPENAI_API_KEY` | — | Required when `BLUE_DOC_EMBEDDING=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model name |
| `BLUE_DOC_REDIS_HOST` | — | Redis host; omit to run without a vector store |
| `BLUE_DOC_REDIS_PORT` | `6379` | Redis port |
| `BLUE_DOC_COLLECTION` | `documents` | Default collection name |
| `BLUE_DOC_CHUNK_SIZE` | `1000` | Max words per chunk |
| `BLUE_DOC_OVERLAP` | `150` | Word overlap between chunks |
| `BLUE_DOC_OCR_FALLBACK` | — | Set to `1` or `true` to enable Tesseract OCR on scanned PDFs |

## Running with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blue-data-doc-source": {
      "command": "python3",
      "args": ["/path/to/blue-data-doc-source/platform/document-mcp/src/server.py"],
      "env": {
        "BLUE_DOC_EMBEDDING": "openai",
        "OPENAI_API_KEY": "sk-...",
        "BLUE_DOC_REDIS_HOST": "localhost",
        "BLUE_DOC_REDIS_PORT": "6379",
        "BLUE_DOC_COLLECTION": "my_docs"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The four tools will appear in the tool picker.

## Running with Claude Code

Add to your project's `.mcp.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "blue-data-doc-source": {
      "command": "python3",
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

## Running standalone

For development or debugging, run the server directly:

```bash
BLUE_DOC_EMBEDDING=dummy python3 platform/document-mcp/src/server.py
```

The server listens on stdio (MCP standard transport). Use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) to call tools interactively:

```bash
npx @modelcontextprotocol/inspector python3 platform/document-mcp/src/server.py
```

## Redis setup

The vector store is optional — omit `BLUE_DOC_REDIS_HOST` to run without it. Ingest still produces chunk records; they just aren't persisted. Search will return a `{"error": ...}` response when no vector source is configured.

To run Redis locally:

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

## Typical agent session

```
Agent: ensure_index(collection="contracts")
→ {"status": "ok", "collection": "contracts"}

Agent: ingest_document(path="/docs/contract.pdf", collection="contracts")
→ {"document_id": "doc_3a7f...", "chunk_count": 42, "file_name": "contract.pdf", "collection": "contracts"}

Agent: search_documents(query="termination clause", collection="contracts", k=3)
→ {"results": [{"chunk_id": "...", "text": "...", "section_title": "Termination", "page": 12}, ...], "count": 3}

Agent: delete_document(document_id="doc_3a7f...", chunk_count=42, collection="contracts")
→ {"deleted": 42}
```

## Running the tests

```bash
python3 -m pytest platform/document-mcp/tests/ -v
```

Tests stub out the MCP SDK and patch the pipeline, so no Redis or OpenAI connection is needed.
