# blue-data-doc-source — Examples

Runnable demos organized in five categories. Most self-contained demos work with no API keys.

---

## Quick start (no API key, no Redis)

```bash
# install the package
pip install -e .

# run the minimal quickstart
python3 examples/backends/memory_quickstart.py
```

---

## Directory layout

```
examples/
  local_memory_store.py        # shared helper: in-process VectorStore (no Redis)
  README.md                    # this file

  scenario-demos/              # Problem → Solution story demos
  feature-demos/               # Drill-down into one feature at a time
  integration-demos/           # End-to-end multi-component workflows
  framework-demos/             # Integration with LangGraph / MCP (OpenAI required)
  backends/                    # Swappable vector store + embedding backends
```

---

## Scenario demos — Problem / Solution stories

| Demo | What it shows | Deps |
|---|---|---|
| `scenario-demos/section_vs_window_demo.py` | SectionChunker vs WindowChunker: fewer cross-topic chunks, cleaner retrieval | none |
| `scenario-demos/provenance_demo.py` | Every search result carries file, page, and section provenance | none |
| `scenario-demos/semantic_search_demo.py` | Semantic paraphrase search vs keyword substring match | OpenAI |

```bash
python3 examples/scenario-demos/section_vs_window_demo.py
python3 examples/scenario-demos/provenance_demo.py --format txt
OPENAI_API_KEY=sk-... python3 examples/scenario-demos/semantic_search_demo.py
```

---

## Feature demos — One feature, deep

| Demo | What it shows | Deps |
|---|---|---|
| `feature-demos/chunking_demo.py` | All three chunkers (SectionChunker, WindowChunker, SentenceChunker); word-count proofs | none |
| `feature-demos/enrichment_demo.py` | BasicEnricher entity types + semantic hints; live document enrichment | none |
| `feature-demos/content_addressing_demo.py` | SHA-256 content-addressed IDs; dedup; stale-chunk detection | none |
| `feature-demos/nl_query_demo.py` | NLQueryParser: search/ingest/delete intent parsing; live execute_query_nl | none |
| `feature-demos/parsers_demo.py` | Raw output of all four parsers (TextParser, PDFParser, DocxParser, PlumberPDFParser) before the pipeline; shows exactly what each parser returns — page dicts, typed blocks, markdown headings, OCR fields | none (OCR: tesseract + poppler; pdfplumber optional) |
| `feature-demos/schema_inference_demo.py` | KeywordClassifier type detection + LLMSchemaExtractor field extraction on four real PDFs; document_type on chunks; DOCUMENT.GET_SCHEMA | none (OpenAI optional) |
| `feature-demos/pdf_formats_demo.py` | PlumberPDFParser structured table extraction vs pypdf flat text; embedded images silently skipped; multi-page docs; real arXiv + Berkshire annual report PDFs; scanned image-only PDFs via Tesseract OCR | pdfplumber (OCR: tesseract + poppler) |
| `feature-demos/rich_search_demo.py` | DOCUMENT.SEARCH_RICH vs plain DOCUMENT.SEARCH side-by-side; provenance block anatomy; neighbor context (context_chunks); by_document grouping with top_sections; in-process caveat | none |

```bash
python3 examples/feature-demos/chunking_demo.py
python3 examples/feature-demos/enrichment_demo.py
python3 examples/feature-demos/content_addressing_demo.py
python3 examples/feature-demos/nl_query_demo.py

# Parser raw output — all four parsers, no API key needed:
python3 examples/feature-demos/parsers_demo.py
# With OCR step (requires: pip install blue-data-doc-source[ocr] && brew install tesseract poppler):
python3 examples/feature-demos/parsers_demo.py

# Schema inference — keyword classification only (no API key needed):
python3 examples/feature-demos/schema_inference_demo.py

# With LLM field extraction:
OPENAI_API_KEY=sk-... python3 examples/feature-demos/schema_inference_demo.py

# PDF formats — tables, images, multi-page, real-world, OCR (no API key needed):
python3 examples/feature-demos/pdf_formats_demo.py
# With OCR (requires: pip install blue-data-doc-source[ocr] && brew install tesseract poppler):
python3 examples/feature-demos/pdf_formats_demo.py

# Rich search — provenance, context, by_document grouping (no API key needed):
python3 examples/feature-demos/rich_search_demo.py
```

---

## Integration demos — End-to-end workflows

| Demo | What it shows | Deps |
|---|---|---|
| `integration-demos/pdf_pipeline_demo.py` | PDF page-scoped chunks; page_start==page_end constraint; provenance comparison vs TXT | none (PDF optional) |
| `integration-demos/multi_format_demo.py` | TXT + PDF + DOCX through one pipeline; ingest_text() and ingest_pages() bypass methods | none |
| `integration-demos/corpus_update_demo.py` | Add / replace / remove documents; DOCUMENT.DELETE; stale-chunk management | none |
| `integration-demos/custom_parser_demo.py` | Parser ABC extension (LlamaIndex wrapper) and ingest_pages() bypass for external extractors | none (LlamaIndex optional) |

```bash
python3 examples/integration-demos/pdf_pipeline_demo.py
python3 examples/integration-demos/multi_format_demo.py
python3 examples/integration-demos/corpus_update_demo.py
python3 examples/integration-demos/custom_parser_demo.py                   # bypass mode, no deps
python3 examples/integration-demos/custom_parser_demo.py --mode llamaindex  # requires llama-index
```

---

## Framework demos — LangGraph and MCP integration

| Demo | What it shows | Deps |
|---|---|---|
| `framework-demos/langgraph_rag_demo.py` | Two-node RAG graph: [retrieve] → [generate] → END | OpenAI, langgraph |
| `framework-demos/mcp_demo.py` | LLM agent using ingest/search/delete MCP tools from document-mcp server | OpenAI, langchain-mcp-adapters, langgraph |

```bash
pip install langgraph langchain-openai
OPENAI_API_KEY=sk-... python3 examples/framework-demos/langgraph_rag_demo.py

# MCP demo also requires the server running:
cd platform/document-mcp
BLUE_DOC_EMBEDDING=dummy BLUE_DOC_REDIS_HOST=localhost python src/server.py
# then in another terminal:
OPENAI_API_KEY=sk-... python3 examples/framework-demos/mcp_demo.py
```

---

## Backend demos — Swappable vector stores and embedding models

| Demo | What it shows | Deps |
|---|---|---|
| `backends/memory_quickstart.py` | Minimal quickstart: LocalMemoryVectorStore + DummyEmbeddingModel | none |
| `backends/redis_backend_demo.py` | RedisVectorSource as persistent backend; DOCUMENT.* op interface | Redis Stack |
| `backends/openai_embeddings_demo.py` | Semantic quality comparison: OpenAIEmbeddingModel vs DummyEmbeddingModel | OpenAI |

```bash
# In-memory (no deps):
python3 examples/backends/memory_quickstart.py

# Redis (requires Redis Stack):
docker run -p 6379:6379 redis/redis-stack-server:latest
python3 examples/backends/redis_backend_demo.py

# OpenAI embeddings:
OPENAI_API_KEY=sk-... python3 examples/backends/openai_embeddings_demo.py
```

---

## Op-code reference

All operations are dispatched via `DocumentDataSource.execute_query(op_dict)`.

| Op | Required fields | Optional | Returns |
|---|---|---|---|
| `DOCUMENT.ENSURE_INDEX` | `collection` | — | `None` |
| `DOCUMENT.INGEST` | `path` | `collection` | `list[dict]` (ChunkRecord) |
| `DOCUMENT.SEARCH` | `query` | `collection`, `k` (default 5), `filter` (dict) | `list[dict]` |
| `DOCUMENT.DELETE` | `document_id`, `chunk_count` OR `path` | `collection` | `None` |
| `DOCUMENT.GET_SCHEMA` | `document_id` | — | `dict` (schema record) |
| `DOCUMENT.SEARCH_RICH` | `query` | `collection`, `k` (default 5), `filter` (dict), `context_chunks` (int, default 1) | `dict` (enriched results with provenance, context, grouping) |

`execute_query_nl(nl_string, collection=...)` parses a natural-language string and dispatches the corresponding op. Requires `nl_parser=NLQueryParser()` at construction.

---

## ChunkRecord field reference

| Field | Type | PDF | TXT/DOCX | Description |
|---|---|---|---|---|
| `document_id` | str | ✓ | ✓ | `"doc_" + sha256(bytes)[:16]` |
| `chunk_id` | str | ✓ | ✓ | `f"{document_id}_chunk_{index:06d}"` |
| `text` | str | ✓ | ✓ | Raw chunk text |
| `file_name` | str | ✓ | ✓ | Source file basename |
| `source_type` | str | ✓ | ✓ | MIME type string |
| `mime_type` | str | ✓ | ✓ | Same as source_type |
| `page_start` | int\|None | ✓ (1-based) | None | First page of chunk |
| `page_end` | int\|None | ✓ (== page_start) | None | Last page of chunk |
| `section_title` | str\|None | ✓ (if detected) | ✓ (if detected) | Header above chunk |
| `document_type` | str\|None | ✓ (if classifier set) | ✓ (if classifier set) | Classified type: contract, invoice, policy, research_paper, financial_report, unknown |
| `content_type` | str\|None | ✓ (if PlumberPDFParser) | ✓ (if PlumberPDFParser) | Content block type: `"text"` or `"table"` |
| `table` | dict\|None | ✓ (if table block) | — | `{"headers": list[str], "rows": list[list[str]]}` — only when `content_type == "table"` |
| `entities` | list[dict] | ✓ | ✓ | BasicEnricher entity spans |
| `semantic_hints` | dict | ✓ | ✓ | word_count, keywords, … |
| `embedding` | list[float]\|None | omitted | omitted | Omitted by `to_dict()` |

`ChunkRecord.to_dict()` omits fields with `None` values — check key presence, not value.

*`pdfplumber` required for `pdf_formats_demo.py` table extraction steps (`pip install pdfplumber`).

---

## Dependency matrix

| Category | numpy | pdfplumber | python-docx | redis-py | openai | langgraph | mcp |
|---|---|---|---|---|---|---|---|
| scenario-demos (no-openai) | ✓ | — | — | — | — | — | — |
| feature-demos | ✓ | optional* | — | — | — | — | — |
| integration-demos | ✓ | optional | — | — | — | — | — |
| framework-demos/langgraph | ✓ | — | — | — | ✓ | ✓ | — |
| framework-demos/mcp | ✓ | — | — | — | ✓ | ✓ | ✓ |
| backends/memory | ✓ | — | — | — | — | — | — |
| backends/redis | ✓ | — | — | ✓ | — | — | — |
| backends/openai | ✓ | — | — | — | ✓ | — | — |
