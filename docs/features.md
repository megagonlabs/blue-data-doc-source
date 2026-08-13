# Features

## Multi-format document parsing

Ingest PDF, plain text, and DOCX files through a single unified interface. The parser is selected automatically by file extension.

- **TXT** — reads the full file as a single text block
- **PDF** — page-by-page extraction via pypdf; optional Tesseract OCR fallback for scanned pages
- **DOCX** — structure-preserving conversion via python-docx; headings and paragraphs mapped to markdown for section detection

```python
from blue_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)

txt_chunks  = pipeline.ingest("report.txt")
pdf_chunks  = pipeline.ingest("contract.pdf")
docx_chunks = pipeline.ingest("policy.docx")

for r in pdf_chunks:
    print(r["file_name"], r["page_start"], r["section_title"], r["text"][:80])
```

---

## Section-aware chunking

`SectionChunker` (the default) detects document structure and keeps each chunk within a single section, preserving the heading as `section_title` on every chunk it produces. When no headers are found it falls back to plain window chunking.

Recognised header patterns:
- Markdown: `# Title`, `## Subtitle`
- Numbered: `1.`, `1.2`, `1.2.3`
- ALL-CAPS lines (≥ 3 words)
- Underlined: `Title\n======`

```python
from blue_doc_source.source.document.chunkers import SectionChunker, WindowChunker

# Default — detects headers, chunks within each section
section_chunker = SectionChunker(chunk_size=1000, overlap=150)
chunks = section_chunker.chunk_with_titles(text)

for title, chunk_text in chunks:
    print(f"§ {title or '(no section)'}")
    print(chunk_text[:100])
```

`WindowChunker` is a word-based sliding window with no structure detection. Use it when documents have no consistent heading style.

```python
window_chunker = WindowChunker(chunk_size=500, overlap=50)
chunks = window_chunker.chunk_with_titles(text)
```

Both chunkers operate on **word count**, not character count. PDF chunks never span page boundaries.

---

## Content-addressed deduplication

Every `ChunkRecord` carries a `document_id` derived from a SHA-256 hash of the source file's bytes. The same file always produces the same `document_id` and the same `chunk_id` values — re-ingesting is a safe upsert that overwrites rather than appends.

```python
from blue_doc_source import DocumentPipeline

pipeline = DocumentPipeline()
records_first  = pipeline.ingest("policy.pdf")
records_second = pipeline.ingest("policy.pdf")  # safe to run again

# IDs are identical — vector store upsert overwrites, no duplicates
assert records_first[0]["document_id"] == records_second[0]["document_id"]
assert records_first[0]["chunk_id"]    == records_second[0]["chunk_id"]
```

`document_id` format: `doc_{sha256(file_bytes)[:16]}`
`chunk_id` format: `{document_id}_chunk_{index:06d}`

---

## Provenance tracking

Every chunk carries a complete provenance chain: the source file, page range (for PDFs), and section heading. Search results are always fully citable without re-reading the original document.

```python
records = pipeline.ingest("annual_report.pdf")

for r in records:
    file    = r["file_name"]                      # "annual_report.pdf"
    page    = r.get("page_start")                 # 1-based; None for TXT/DOCX
    section = r.get("section_title", "(none)")    # heading detected by SectionChunker
    print(f"{file}  page {page}  § {section}")
    print(f"  {r['text'][:120]}")
```

For PDFs, `page_start == page_end` — V1 chunks never span page boundaries. `section_title` is omitted when no heading is detected (`ChunkRecord.to_dict()` omits `None` fields).

---

## Regex enrichment

`BasicEnricher` extracts structured metadata from chunk text using regular expressions — no ML dependency, no API key, zero latency.

Extracted entity types:
- **DATE** — `January 2026`, `2026-03-15`, `15/03/26`
- **EMAIL** — `user@example.com`
- **PERCENT** — `12.5%`, `50 percent`
- **MONEY** — `$1,500`, `€200`, `£50`
- **PROPER_NOUN** — capitalised multi-word phrases

```python
from blue_doc_source.source.document.enrichment import BasicEnricher

enricher = BasicEnricher()
result = enricher.enrich(
    "The deductible of $1,500 applies from January 2026. Contact hr@company.com for details."
)

print(result["entities"])
# [{"text": "$1,500", "type": "MONEY"},
#  {"text": "January 2026", "type": "DATE"},
#  {"text": "hr@company.com", "type": "EMAIL"}]

print(result["semantic_hints"])
# {"word_count": 15, "sentence_count": 2, "has_numbers": True,
#  "has_questions": False, "keywords": ["deductible", "contact", "details"]}
```

Pass `enricher=BasicEnricher()` to `DocumentDataSource` to attach enrichment to every ingested chunk.

---

## Pluggable embedding models

`EmbeddingModel` is an ABC with a single `embed_text(text) -> list[float]` method and a `dim` property. Swap implementations without changing the pipeline.

```python
from blue_doc_source import OpenAIEmbeddingModel, DummyEmbeddingModel

# Production — calls OpenAI text-embedding-3-small (1536-dim)
openai_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
print(openai_model.dim)   # 1536

# Tests — SHA-256 hash → fixed-dim float vector (not real similarity)
dummy_model = DummyEmbeddingModel()
print(dummy_model.dim)    # 8
```

Bring your own model by subclassing `EmbeddingModel`:

```python
from blue_doc_source.source.document.embeddings import EmbeddingModel

class MyEmbeddingModel(EmbeddingModel):
    @property
    def dim(self) -> int:
        return 768

    def embed_text(self, text: str) -> list[float]:
        return my_inference_server.embed(text)
```

Supported OpenAI models: `text-embedding-3-small` (1536), `text-embedding-3-large` (3072), `text-embedding-ada-002` (1536).

---

## Vector storage

`DocumentDataSource` and `DocumentPipeline` accept any `VectorSource` from blue-data. Upsert fires only when both `embedding_model` and `vector_source` are set.

```python
from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
from blue_doc_source import DocumentDataSource, OpenAIEmbeddingModel

# Dev / test — in-process, no external services
source = DocumentDataSource(
    name="docs",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    default_collection="reports",
)

# Production — persistent Redis Stack
source = DocumentDataSource(
    name="docs",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=RedisVectorSource(name="redis", host="localhost", port=6379),
    default_collection="reports",
)

# Create index before first ingest
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "reports"})
```

`DOCUMENT.ENSURE_INDEX` is safe to call multiple times — skip creation if the index already exists.

---

## Natural language queries

`NLQueryParser` translates plain-English instructions into `DOCUMENT.*` op dicts using an LLM, then dispatches the resulting op automatically.

```python
from blue_doc_source import DocumentDataSource, NLQueryParser, OpenAIEmbeddingModel

source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=vector_store,
    nl_parser=NLQueryParser(),          # uses GPT-4o-mini by default
)

# NLQueryParser maps these to DOCUMENT.SEARCH, DOCUMENT.INGEST, DOCUMENT.DELETE
results = source.execute_query_nl("find the termination clause")
records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")
source.execute_query_nl("delete report.txt from reports")
```

Inject any `(str) -> dict` callable to use a different LLM or run without an API key:

```python
import re

def _regex_parse(text: str) -> dict:
    if re.search(r"\b(search|find|look)\b", text, re.I):
        return {"op": "DOCUMENT.SEARCH", "query": text, "k": 5}
    return {"op": "DOCUMENT.INGEST", "path": text}

parser = NLQueryParser(parse_fn=_regex_parse)
```

---

## Scanned PDF support (OCR)

By default, `PDFParser` reads the embedded text layer. Pages with no text — scanned documents, photographed pages — are silently skipped. Pass `ocr_fallback=True` to run Tesseract on any blank page instead.

```python
from blue_doc_source.source.document.parsers import PDFParser

parser = PDFParser(ocr_fallback=True)
pages = parser.parse_pages("scanned_contract.pdf")

for p in pages:
    print(p["page_num"], p.get("ocr"), p["text"][:80])
    # ocr=True marks Tesseract-extracted pages
```

Install OCR dependencies:

```bash
pip install blue-data-doc-source[ocr]   # pytesseract + pdf2image
brew install tesseract                   # macOS; Tesseract must be on PATH
brew install poppler                     # macOS; required by pdf2image
```

A clear `ImportError` is raised if OCR is requested but dependencies are missing.

---

## Pluggable parsers

The built-in parsers cover standard digitally-created files. To use an external document processor — LlamaIndex, a vision model, a cloud OCR API — implement the `Parser` ABC and pass it to `DocumentPipeline`. All downstream stages (chunking, enrichment, embedding, storage) are unchanged.

```python
from blue_doc_source import DocumentPipeline, Parser

class LlamaIndexParser(Parser):
    def parse(self, path: str):
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return [{"text": doc.text}]

pipeline = DocumentPipeline(parser=LlamaIndexParser(), embedding_model=...)
pipeline.ingest("complex_layout.pdf")
```

`parse()` must return `list[dict]` with `"text": str`. Include `"page_num": int` (1-based) for page-aware provenance.

Bypass the parser entirely when text is already extracted:

```python
# Pre-extracted text
pipeline.ingest_text(text, file_name="report.txt", mime_type="text/plain")

# Pre-extracted pages
pipeline.ingest_pages(
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}],
    file_name="contract.pdf",
    mime_type="application/pdf",
)
```

---

## MCP server

`platform/document-mcp` exposes `DocumentDataSource` as tools for Claude and other MCP clients. Each tool maps to a `DOCUMENT.*` op and returns structured JSON.

```bash
# Start with dummy embeddings (no OpenAI required)
cd platform/document-mcp
BLUE_DOC_EMBEDDING=dummy python src/server.py

# Production: OpenAI embeddings + Redis
BLUE_DOC_EMBEDDING=openai \
OPENAI_API_KEY=sk-... \
BLUE_DOC_REDIS_HOST=localhost \
python src/server.py
```

**Claude Desktop config:**

```json
{
  "mcpServers": {
    "blue-doc": {
      "command": "python",
      "args": ["/path/to/platform/document-mcp/src/server.py"],
      "env": {
        "BLUE_DOC_EMBEDDING": "openai",
        "OPENAI_API_KEY": "sk-...",
        "BLUE_DOC_REDIS_HOST": "localhost"
      }
    }
  }
}
```

Tools exposed: `ensure_index`, `ingest_document`, `ingest_documents`, `search`, `delete_document`.

See [`docs/mcp.md`](mcp.md) for the full tool reference and environment variable documentation.
