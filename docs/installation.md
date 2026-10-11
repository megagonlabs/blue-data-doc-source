# Installation

## Components

`blue-data-doc-source` ships four installable packages:

| Component | Location | Description |
|---|---|---|
| `blue-data-doc-source` | `lib/` | Core Python library — parsing, chunking, enrichment, embeddings, pipeline |
| `blue-data-doc-api` | `platform/doc-api/` | FastAPI REST server — HTTP interface with interactive docs at `/docs` |
| `blue-data-doc-cli` | `platform/doc-cli/` | CLI — ingest, search, and delete documents from a terminal |
| `blue-data-doc-mcp` | `platform/doc-mcp/` | MCP server — exposes all pipeline ops as tools for Claude Desktop |

---

## Installing the library

`blue-data-doc-source` depends on `blue-data` (a sibling package in the same monorepo). Install it first, then install this package:

```bash
pip install -e ../blue-data
pip install -e lib/
```

To install with optional extras:

```bash
pip install -e ".[ocr]"   # Tesseract OCR support for scanned PDFs
pip install -e ".[mcp]"   # MCP server support (platform/doc-mcp)
pip install -e ".[nlp]"   # NLTK-based processing
```

## Installing the REST API

```bash
pip install -e lib/
pip install -e platform/doc-api/

blue-data-doc-api          # starts on http://localhost:8001
```

## Installing the CLI

```bash
pip install -e lib/
pip install -e platform/doc-cli/

doc --help
```

## Installing the MCP server

```bash
pip install -e lib/
pip install -e ".[mcp]"

python platform/doc-mcp/src/server.py
```

---

## Runtime dependencies

Installed automatically with `pip install`:

| Package | Required for |
|---|---|
| `openai` | `OpenAIEmbeddingModel`, `NLQueryParser` |
| `pypdf` | `PDFParser` — reads text layer from digitally-created PDFs |
| `python-docx` | `DocxParser` — reads `.docx` files |
| `redis[hiredis]` | `RedisVectorSource` (from `blue-data`) for production vector storage |

Optional extras:

| Extra | Packages | Required for |
|---|---|---|
| `[ocr]` | `pytesseract`, `pdf2image` | OCR fallback for scanned PDFs with no text layer |
| `[mcp]` | `mcp` | Running the `document-mcp` MCP server |
| `[nlp]` | `nltk` | NLTK-based text processing |

---

## External dependencies

**Redis Stack** — required for production vector storage (`RedisVectorSource`). Standard Redis does not support the `FT.CREATE` and `FT.SEARCH` commands used by `RedisVectorSource`.

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

**Tesseract** — required when using `PDFParser(ocr_fallback=True)` on scanned PDFs:

```bash
brew install tesseract          # macOS
apt-get install tesseract-ocr   # Ubuntu/Debian
```

**Poppler** — required by `pdf2image` (part of the `[ocr]` extra):

```bash
brew install poppler            # macOS
apt-get install poppler-utils   # Ubuntu/Debian
```

**OpenAI API key** — required for `OpenAIEmbeddingModel` and `NLQueryParser`:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Running tests

```bash
# Unit tests — no external services required
pytest lib/tests/unit/ -v

# Integration tests — require Redis Stack on localhost:6379
docker run -d -p 6379:6379 redis/redis-stack-server:latest
pytest lib/tests/integration/ -v -m integration

# All tests
pytest lib/tests/ -v
```

Install the test dependencies first (includes `pytest`, `fpdf2`, and `pdfplumber`, which the PDF/table tests require):

```bash
pip install -e "lib/.[dev]"
```
