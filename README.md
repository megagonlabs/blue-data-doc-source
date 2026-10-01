# blue-data-doc-source

Turn PDFs, DOCX, and text files into a queryable, filterable blue-data source.

Documents are data — not merely search-index content. `blue-data-doc-source` handles the full path from raw file to queryable vector index, and plugs into blue-data's `DataRegistry` so document evidence becomes a first-class member of the same intelligence layer as your SQL tables, graph stores, and web content.

→ **[Full documentation](docs/README.md)** · **[Quick start](docs/quickstart.md)** · **[Installation](docs/installation.md)** · **[Examples](examples/README.md)** · **[CLI](docs/cli.md)** · **[MCP server](docs/mcp.md)**

---

## What it does

- **Ingest** PDF, DOCX, and text files — parse, classify, chunk, enrich, embed, and upsert in one call.
- **Classify** every document by type at zero API cost and propagate it to every chunk for pre-filtered search.
- **Extract** typed fields at ingest (parties, totals, dates) via an LLM or a caller-supplied schema, then filter on them deterministically with `DOCUMENT.FIND` — no embedding, no query-time LLM.
- **Search** by semantic similarity with metadata pre-filtering, or combine BM25 + vector in one op.
- **Navigate** an optional knowledge graph of documents, sections, chunks, and entities for exact entity lookup.
- **Register** documents in blue-data's `DataRegistry` alongside SQL, web, and graph sources, with a consistent chunk schema.

Full feature list: [docs/features.md](docs/features.md). Runnable demos: [examples/](examples/).

---

## Repository layout

```
blue-data-doc-source/
├── lib/                    # Core Python library (blue-data-doc-source)
│   ├── src/blue_data_doc_source/
│   └── tests/              # unit, integration, and benchmark tests
├── platform/
│   ├── doc-cli/            # `blue-docdb` command-line interface
│   ├── doc-mcp/            # MCP server (Claude Desktop / Claude Code)
│   └── doc-api/            # FastAPI REST service (port 8001)
├── docs/                   # Documentation
├── examples/               # Runnable example scripts with captured output
├── design/                 # Design notes and decision log
└── data/                   # Sample documents used by the examples
```

---

## Requirements

- Python 3.10+
- Sibling repo cloned beside this one: [`blue-data`](../blue-data)
- `OPENAI_API_KEY` — for OpenAI embeddings, LLM field extraction/enrichment, and natural-language queries (not needed for parsing, classification, or dummy embeddings):

```bash
export OPENAI_API_KEY=sk-...
```

Optional extras (`[ocr]`, `[nlp]`, `[mcp]`) are documented in [docs/installation.md](docs/installation.md).

---

## Quick start

```bash
# Clone blue-data beside this repo first, then:
python -m venv .venv && source .venv/bin/activate
bash lib/scripts/install.sh          # installs blue-data and lib/

# Ingest a document and inspect the extracted chunks — no API key required:
python - <<'PY'
from blue_data_doc_source import DocumentPipeline

for r in DocumentPipeline(chunk_size=500, overlap=100).ingest("data/sample_policy.txt"):
    print(r["chunk_id"], r.get("page_start"), r["text"][:80])
PY
```

For the full end-to-end walkthrough — embeddings, a vector store, graph indexing, and search — see [docs/quickstart.md](docs/quickstart.md) and the [guide](docs/guide.md).

---

## Packages

| Package | Description |
|---|---|
| [`lib/`](lib/README.md) | Core library `blue-data-doc-source` — `DocumentDataSource`, `DocumentPipeline`, and all ingestion components. Install: `bash lib/scripts/install.sh` |
| [`platform/doc-cli/`](platform/doc-cli/README.md) | `blue-docdb` command-line interface over the core ops. Install: `pip install -e platform/doc-cli` |
| [`platform/doc-mcp/`](platform/doc-mcp/README.md) | MCP server exposing document ops as AI-callable tools. Install: `pip install -e platform/doc-mcp` |
| [`platform/doc-api/`](platform/doc-api/README.md) | FastAPI REST service on port 8001. Install: `pip install -e platform/doc-api` |
