# Installation

## Components

`blue-data-doc-source` ships a single installable package that covers the core library and the MCP platform component.

| Component | Location | Description |
|---|---|---|
| `blue-data-doc-source` | `lib/` | Core Python library — parsing, chunking, enrichment, embeddings, pipeline |
| `document-mcp` | `platform/document-mcp/` | MCP server — exposes all pipeline ops as tools for Claude Desktop |

---

## Installing the library

`blue-data-doc-source` depends on [`blue-data`](../blue-data). Install it first, then install this package:

```bash
pip install -e ../blue-data
pip install -e .
```

To install with optional extras:

```bash
pip install -e ".[ocr]"   # Tesseract OCR support for scanned PDFs
pip install -e ".[mcp]"   # MCP server support (platform/document-mcp)
pip install -e ".[nlp]"   # NLTK-based processing
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

Install dev dependencies before running tests:

```bash
pip install pytest pytest-httpx
```
