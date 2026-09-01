# blue-data-doc-source — High Level Architecture

## Diagram

```
                    External Interfaces

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Python Lib  │  │ MCP Server  │  │  REST API   │  │     CLI     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┴────────────────┴────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DocumentDataSource                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       DocumentPipeline                      │   │
│  │                                                             │   │
│  │  ┌──────────┐   ┌────────────────────────────────────────┐  │   │
│  │  │  Parser  │ → │           Schema Layer                 │  │   │
│  │  │          │   │  DocumentClassifier + FieldExtractor   │  │   │
│  │  │ .txt     │   │  RecordStore                           │  │   │
│  │  │ .pdf     │   └────────────────────────────────────────┘  │   │
│  │  │ .docx    │                    │                           │   │
│  │  └──────────┘                    ▼                           │   │
│  │                   ┌──────────────────────────┐               │   │
│  │                   │        Chunker           │               │   │
│  │                   │  SectionChunker (default)│               │   │
│  │                   │  WindowChunker           │               │   │
│  │                   │  SentenceChunker         │               │   │
│  │                   └──────────────────────────┘               │   │
│  │                                  │                           │   │
│  │                                  ▼                           │   │
│  │                   ┌──────────────────────────┐               │   │
│  │                   │        Enricher          │               │   │
│  │                   │  BasicEnricher (default) │               │   │
│  │                   │  LLMEnricher             │               │   │
│  │                   └──────────────────────────┘               │   │
│  │                                  │                           │   │
│  │                                  ▼                           │   │
│  │                   ┌──────────────────────────┐               │   │
│  │                   │     EmbeddingModel       │               │   │
│  │                   │  OpenAIEmbeddingModel    │               │   │
│  │                   │  DummyEmbeddingModel     │               │   │
│  │                   └──────────────────────────┘               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │       GraphIndexer           │  │      NL Interface          │  │
│  │  (optional; graph_source≠∅) │  │   NLQueryParser (Claude)   │  │
│  └──────────────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────┐
           ▼                              ▼                          ▼
┌──────────────────┐          ┌────────────────────┐    ┌──────────────────┐
│   VectorSource   │          │    RecordStore      │    │   GraphSource    │
│  ┌────────────┐  │          │  ┌──────────────┐  │    │  ┌────────────┐  │
│  │   Memory   │  │          │  │    Memory    │  │    │  │   Memory   │  │
│  │   Vector   │  │          │  │    Record    │  │    │  │   Graph    │  │
│  └────────────┘  │          │  └──────────────┘  │    │  └────────────┘  │
│  ┌────────────┐  │          │  ┌──────────────┐  │    │  ┌────────────┐  │
│  │   Redis    │  │          │  │     File     │  │    │  │   Neo4j    │  │
│  │   Vector   │  │          │  │    Record    │  │    │  │   Graph    │  │
│  └────────────┘  │          │  └──────────────┘  │    │  └────────────┘  │
└──────────────────┘          │  ┌──────────────┐  │    └──────────────────┘
                              │  │  JSONSource  │  │
                              │  │   Record     │  │
                              │  └──────────────┘  │
                              └────────────────────┘
```

## Components

### External Interfaces

Four interface adapters sit in front of `DocumentDataSource` and delegate to it. They share no business logic — each is a thin translation layer.

**Python Library** — Native Python API (`DocumentDataSource`, `DocumentPipeline`). Used to embed document ingestion and retrieval directly in Python applications and agentic pipelines.

**MCP Server** — Exposes `ensure_index`, `ingest_document`, `search_documents`, `find_documents`, `find_then_search`, `get_document_record`, and `delete_document` as MCP tools. Enables LLM agents to ingest and search documents within an agentic context.

**REST API** — HTTP/JSON interface (FastAPI, port 8001). Routes under `/documents/` mirror the nine op codes. No business logic in handlers — all calls delegate to `DocumentPipeline`.

**CLI** — Click command group (`blue-data-doc`). Commands: `ensure-index`, `ingest`, `search`, `delete`.

---

### DocumentDataSource

The top-level public entry point. Extends blue-data's `DataSource` ABC. Owns a single `DocumentPipeline` instance and dispatches on nine op codes: `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.INGEST`, `DOCUMENT.SEARCH`, `DOCUMENT.FIND`, `DOCUMENT.FIND_THEN_SEARCH`, `DOCUMENT.GET_RECORD`, `DOCUMENT.DELETE`, `DOCUMENT.GRAPH_NEIGHBORS`, `DOCUMENT.GRAPH_QUERY`.

---

### DocumentPipeline

Orchestrates the full ingestion chain. Auto-detects parser by file extension (`.txt`, `.pdf`, `.docx`). Owns all pipeline components and wires them in sequence. Key methods: `ingest(path)`, `search(query)`, `find_records(filter)`, `ensure_index(collection)`, `delete(path)`, `graph_neighbors(...)`, `graph_query(...)`.

---

### Parser

Converts file bytes to a list of page dicts `[{"text": str, "page_num"?: int}]`. Four implementations:

| Class | Format | Notes |
|-------|--------|-------|
| `TextParser` | `.txt` | stdlib only; UTF-8 |
| `PDFParser` | `.pdf` | `pypdf`; OCR fallback via `pytesseract` + `pdf2image` when `[ocr]` extra installed |
| `PlumberPDFParser` | `.pdf` | `pdfplumber`; table-aware; returns typed blocks with `"type": "table"` |
| `DocxParser` | `.docx` | `python-docx`; preserves heading structure for `SectionChunker` |

---

### Schema Layer

Runs after parsing, before chunking.

**DocumentClassifier** scores the first 500 words of the document against signal phrase sets (`KeywordClassifier`, zero API cost) or calls an LLM (`LLMFieldExtractor`) to detect document type (`contract`, `invoice`, `policy`, `research`, `unknown`) with a confidence score.

**FieldExtractor** extracts type-specific fields: `parties`, `governing_law`, `effective_date` for contracts; `vendor`, `total_due`, `line_items` for invoices; etc. Three implementations: `LLMFieldExtractor` (one GPT-4o-mini call per document), `StructuredFieldExtractor` (caller-supplied Pydantic / TypedDict / JSON Schema), `KeywordClassifier` (zero-cost type detection only, no field extraction).

**RecordStore** persists the `DocumentRecord` (`{document_id, document_type, confidence, classifier, fields}`) keyed by `document_id`. Three backends: `MemoryRecordStore`, `FileRecordStore`, `JsonSourceRecordStore` (any blue-data `JSONSource`).

---

### Chunker

Splits extracted text into `(section_title, text)` pairs. Three implementations, all operating on **word count** (not character count):

| Class | Default? | Strategy |
|-------|----------|---------|
| `SectionChunker` | Yes | Detects Markdown/heading markers; groups by section; falls back to window when no headings |
| `WindowChunker` | No | Sliding word-count window with configurable `chunk_size` and `overlap` |
| `SentenceChunker` | No | Sentence-boundary aware; requires `[nlp]` extra (`nltk`) |

---

### Enricher

Annotates each chunk text with entities and semantic hints. Two implementations:

| Class | Cost | Output |
|-------|------|--------|
| `BasicEnricher` | Zero (regex) | `entities`: PROPER_NOUN matches (2+ title-case words); `semantic_hints`: `has_numeric`, `has_date`, `is_short` |
| `LLMEnricher` | One GPT-4o-mini call per chunk | `entities`: typed list (PERSON, ORG, LOCATION, DATE); `keyphrases`, `relationships`, semantic flags |

---

### EmbeddingModel

Converts chunk text to a fixed-dimension float vector. Two implementations:

| Class | Notes |
|-------|-------|
| `OpenAIEmbeddingModel` | `text-embedding-3-small`; dim=1536; one API call per chunk |
| `DummyEmbeddingModel` | SHA-256 hash → deterministic vector; for dev/test; no API cost |

The `dim` property is required on all implementations — used by `DocumentPipeline` to configure the vector index at `DOCUMENT.ENSURE_INDEX` time and validated against each embedding on the first chunk of every ingest.

---

### GraphIndexer

Optional side-effect component. When `graph_source` is set, writes a typed graph during `DOCUMENT.INGEST`: `Document`, `Section`, `Chunk`, and `Entity` nodes; eight edge types (`HAS_SECTION`, `HAS_CHUNK`, `NEXT`, `PREV`, `MENTIONS`, `CO_OCCURS_WITH`, `APPEARS_IN`). Entity nodes are shared across documents and never removed by `DOCUMENT.DELETE`. When `graph_source=None` (the default), `GraphIndexer` is a no-op — zero overhead.

---

### NL Interface

`NLQueryParser` converts free-text queries to `DOCUMENT.*` op dicts using Claude's tool-use API. Requires the `anthropic` package and `ANTHROPIC_API_KEY`.

---

### VectorSource

Provided by `blue-data`. `DocumentPipeline` interacts only with the `VectorSource` interface — no adapter layer. Two backends: `MemoryVectorSource` (in-process numpy cosine; dev/test) and `RedisVectorSource` (production; requires Redis Stack).

---

### RecordStore

Persists `DocumentRecord` schema classification results. Three backends: `MemoryRecordStore` (default; in-process), `FileRecordStore` (atomic JSON; survives restarts), `JsonSourceRecordStore` (any blue-data `JSONSource`; multi-process safe).

---

### GraphSource

Provided by `blue-data`. Optional — `graph_source=None` disables all graph functionality. Two backends: `MemoryGraphSource` (dev/test) and `Neo4JGraphSource` (production).
