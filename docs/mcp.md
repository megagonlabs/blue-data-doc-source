# MCP Server

The MCP server exposes blue-data-doc-source as AI-callable tools over the Model Context Protocol. Any MCP-compatible agent — Claude Desktop, Claude Code, or a custom agent using the MCP SDK — can ingest documents, search them, filter by extracted fields, retrieve records, traverse the entity graph, and manage the vector index without any custom integration code.

## Tools

| Tool | Op equivalent | Description |
|---|---|---|
| `ensure_index` | `DOCUMENT.ENSURE_INDEX` | Create the vector index before first ingest |
| `ingest_document` | `DOCUMENT.INGEST` | Parse, chunk, enrich, embed, and upsert a document |
| `search_documents` | `DOCUMENT.SEARCH` | Semantic search over ingested documents |
| `find_documents` | `DOCUMENT.FIND` | Filter documents by extracted fields — deterministic, no embedding/LLM |
| `find_then_search` | `DOCUMENT.FIND_THEN_SEARCH` | Filter documents by fields, then search text within the matches |
| `get_document_record` | `DOCUMENT.GET_RECORD` | Retrieve a document's structured record (type, confidence, extracted fields) |
| `graph_neighbors` | `DOCUMENT.GRAPH_NEIGHBORS` | Find chunks mentioning an entity via the graph (requires `BLUE_DOC_GRAPH=1`) |
| `graph_query` | `DOCUMENT.GRAPH_QUERY` | Traverse the entity/structure graph from any node (requires `BLUE_DOC_GRAPH=1`) |
| `delete_document` | `DOCUMENT.DELETE` | Remove all chunks for a document from the vector store |

`find_documents` / `get_document_record` work without a vector store (they read the per-document record store); `find_then_search` and `search_documents` require one. The graph tools require `BLUE_DOC_GRAPH=1`.

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
| `BLUE_DOC_GRAPH` | — | Set to `1` or `true` to enable the entity graph (`graph_neighbors` / `graph_query` tools) |

## Running with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blue-data-doc-source": {
      "command": "python3",
      "args": ["/path/to/blue-data-doc-source/platform/doc-mcp/src/server.py"],
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

Restart Claude Desktop after saving. The tools will appear in the tool picker.

## Running with Claude Code

Add to your project's `.mcp.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "blue-data-doc-source": {
      "command": "python3",
      "args": ["platform/doc-mcp/src/server.py"],
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
BLUE_DOC_EMBEDDING=dummy python3 platform/doc-mcp/src/server.py
```

The server listens on stdio (MCP standard transport). Use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) to call tools interactively:

```bash
npx @modelcontextprotocol/inspector python3 platform/doc-mcp/src/server.py
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
python3 -m pytest platform/doc-mcp/tests/ -v
```

Tests stub out the MCP SDK and patch the pipeline, so no Redis or OpenAI connection is needed.
