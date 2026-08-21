# blue-data-doc-source — Deployment & Ops

## Packages

`blue-data-doc-source` ships as three separate installable artifacts:

| Artifact | PyPI name | Entry point |
|----------|-----------|-------------|
| Core library | `blue-data-doc-source` | Python import |
| REST API | `blue-data-doc-api` | `uvicorn doc_api.main:app` |
| CLI | `blue-data-doc-cli` | `blue-data-doc` |

The MCP server (`platform/document-mcp/`) is not published separately — it is run directly via `python -m server` or invoked by an MCP host over stdio.

---

## External Dependencies

| Service | Required by | Notes |
|---------|------------|-------|
| Redis Stack | REST API, MCP, CLI (production) | `redis/redis-stack-server:latest`; provides both JSON (`ReJSON`) and vector search (`RediSearch`) modules; port 6379 |
| OpenAI API | REST API, MCP, CLI (when `BLUE_DOC_EMBEDDING=openai`) | `text-embedding-3-small`; also used by `LLMEnricher` and `LLMFieldExtractor` when configured |
| Tesseract | Optional (`BLUE_DOC_OCR_FALLBACK=1`) | System binary; must be installed separately alongside `[ocr]` Python extras |

Without `BLUE_DOC_REDIS_HOST`, the pipeline falls back to `MemoryVectorSource` — all data is lost on restart. This mode is suitable for local development only.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BLUE_DOC_EMBEDDING` | No | `dummy` | Embedding model: `openai` or `dummy` (SHA-256 hash; for dev/test only) |
| `OPENAI_API_KEY` | When `BLUE_DOC_EMBEDDING=openai` | — | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model name passed to the OpenAI embeddings API |
| `BLUE_DOC_REDIS_HOST` | No | — | Redis host; omit to use `MemoryVectorSource` |
| `BLUE_DOC_REDIS_PORT` | No | `6379` | Redis port |
| `BLUE_DOC_COLLECTION` | No | `documents` | Default vector index collection name |
| `BLUE_DOC_CHUNK_SIZE` | No | `1000` | Word-count target per chunk (passed to `SectionChunker` / `WindowChunker`) |
| `BLUE_DOC_OVERLAP` | No | `150` | Word-count overlap between consecutive window chunks |
| `BLUE_DOC_OCR_FALLBACK` | No | — | Set to `1` or `true` to enable Tesseract OCR for scanned PDFs |

All variables are read once at startup during `DocumentPipeline` singleton construction. Changing them requires a process restart.

---

## Running the REST API

### Development

```bash
cd platform/api
pip install -e ".[dev]"
BLUE_DOC_REDIS_HOST=localhost BLUE_DOC_EMBEDDING=openai OPENAI_API_KEY=sk-... \
  uvicorn doc_api.main:app --reload --host 0.0.0.0 --port 8001
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e "lib/[openai]" && pip install -e "platform/api/"
EXPOSE 8001
CMD ["uvicorn", "doc_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

```bash
docker run -p 8001:8001 \
  -e BLUE_DOC_REDIS_HOST=redis \
  -e BLUE_DOC_EMBEDDING=openai \
  -e OPENAI_API_KEY=sk-... \
  blue-data-doc-api
```

### Redis Stack

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

---

## Running the MCP Server

The MCP server communicates over **stdio** — it is launched by the MCP host (e.g. Claude Desktop) rather than run as a daemon.

```bash
# Direct invocation (for testing)
BLUE_DOC_REDIS_HOST=localhost BLUE_DOC_EMBEDDING=openai OPENAI_API_KEY=sk-... \
  python platform/document-mcp/src/server.py
```

MCP host config entry (e.g. `claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "blue-data-doc": {
            "command": "python",
            "args": ["/path/to/blue-data-doc-source/platform/document-mcp/src/server.py"],
            "env": {
                "BLUE_DOC_REDIS_HOST": "localhost",
                "BLUE_DOC_EMBEDDING": "openai",
                "OPENAI_API_KEY": "sk-..."
            }
        }
    }
}
```

---

## Running the CLI

```bash
pip install -e "lib/" -e "platform/cli/"

export BLUE_DOC_REDIS_HOST=localhost
export BLUE_DOC_EMBEDDING=openai
export OPENAI_API_KEY=sk-...

blue-data-doc ensure-index
blue-data-doc ingest /path/to/report.pdf
blue-data-doc search "eligibility requirements" -k 5
blue-data-doc delete --path /path/to/report.pdf
```

---

## Port Assignments

| Service | Port |
|---------|------|
| blue-data REST API | 8000 |
| **blue-data-doc-source REST API** | **8001** |
| blue-data-web-source REST API | 8002 |
| Redis Stack | 6379 |

---

## Index Lifecycle

1. Call `DOCUMENT.ENSURE_INDEX` (or `blue-data-doc ensure-index`) before the first ingest. The REST API does **not** auto-create the index on startup — this is intentional so deployments can manage index schema explicitly.
2. `ENSURE_INDEX` is idempotent — safe to call on every deploy.
3. Changing `BLUE_DOC_CHUNK_SIZE` or the embedding model dimension after initial indexing requires dropping and recreating the index. Re-ingest all documents after an index schema change.

---

## Observability

No OTEL or Prometheus instrumentation is wired in V1. Observability relies on:

- **Uvicorn access logs** — request/response logging from the ASGI server.
- **Startup log** — the REST API logs the resolved pipeline configuration (embedding model, Redis host, collection, chunk size) at startup.
- **Error propagation** — unhandled exceptions return `500` with an error body; `404` for missing records.

Structured logging and metrics are planned for a future release.

---

## Tesseract OCR

When `BLUE_DOC_OCR_FALLBACK=1`, `PDFParser` falls back to Tesseract for pages where `pypdf` returns near-empty text. This requires:

1. Tesseract binary installed on the host (`apt-get install tesseract-ocr` on Debian/Ubuntu).
2. Python extras: `pip install "blue-data-doc-source[ocr]"` (installs `pytesseract` and `pdf2image`).
3. `poppler-utils` for `pdf2image` (`apt-get install poppler-utils`).

OCR is CPU-intensive and significantly slower than direct text extraction. It is disabled by default.
