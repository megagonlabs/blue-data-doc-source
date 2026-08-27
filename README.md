# blue-data-doc-source

Documents are data — not merely search-index content. blue-data-doc-source makes them queryable and filterable alongside SQL tables, graph stores, JSON sources, and knowledge bases. All are registered through the same blue-data `DataRegistry`, allowing agents to retrieve document evidence and cross-reference it with other enterprise data sources.

It handles the full pipeline from raw file to queryable data source: parsing PDFs, DOCX, and text files; assigning document type and extracting typed fields at ingest; chunking with section awareness; enriching with entities and keyphrases; embedding; and upserting into blue-data's vector layer. The result is a `DocumentDataSource` — a first-class member of the blue-data intelligence layer, not a standalone search box.

→ **[Full feature list](docs/features.md)** · **[Installation](docs/installation.md)** · **[Guide](docs/guide.md)** · **[Extending](docs/extending.md)** · **[Examples](examples/README.md)** · **[MCP server](docs/mcp.md)**

## Why blue-data-doc-source

The genuine differentiator is not any single feature — it is the combination: document-level typed records, chunk-level semantic retrieval, a two-stage find-then-search query pattern, and integration with blue-data's broader data-source and registry model. Together these make documents a structured, queryable part of an agent's data layer rather than a separate search silo.

**Failure 1 — Semantic search returns the wrong document**

A corpus of 100 vendor contracts. The agent asks *"what does Acme Corp's termination clause say?"* A conventional vector-search pipeline embeds the query and retrieves the most semantically similar chunks across the corpus. Without document-level filtering, chunks from unrelated contracts may outrank the relevant Acme Corp clause — particularly when that contract uses "cancellation" rather than "termination."

blue-data-doc-source adds a two-step pattern. At ingest time, `LLMFieldExtractor` extracts `parties`, `governing_law`, and `effective_date` from each contract — one API call per document, not per query. At query time, `DOCUMENT.FIND` evaluates filters over the persisted extracted fields, then `DOCUMENT.SEARCH` retrieves the clause within the matched document only.

```python
# Step 1 — structured filter: no embedding, no LLM, deterministic
docs = pipeline.find_records({"fields.parties": {"contains": "Acme Corp"}})

# Step 2 — text retrieval scoped to the matched document only
chunks = pipeline.search("termination clause", filters={"document_id": docs[0]["document_id"]})
# → ['2. TERMINATION — Either party may terminate this agreement with sixty days...']
```

**Failure 2 — Structured questions can't be answered by embedding distance**

*"Find all invoices from Blue Solutions where the total exceeds $30,000."* A vector search returns chunks that *mention* Blue Solutions and amounts. It cannot evaluate `total_due > 30000` — that requires parsing `"$35,154"` as a number, which embedding distance cannot do. The answer is either wrong or requires the LLM to read every invoice chunk and compare manually.

blue-data-doc-source extracts `vendor` and `total_due` from each invoice at ingest. The query becomes a direct filter — no query-time LLM comparison, no embedding cost. (Field values are extracted by an LLM at ingest; the filter evaluation itself is deterministic over those persisted values.)

```python
pipeline.find_records({
    "document_type": "invoice",
    "fields.vendor":    {"contains": "Blue Solutions"},
    "fields.total_due": {"gt": 30000},   # "$35,154" → 35154 → 35154 > 30000 ✓
})
# → [{document_type: "invoice", fields: {total_due: "$35,154", vendor: "Blue Solutions Inc.", ...}}]
```

**Failure 3 — A mixed corpus makes every search ambiguous**

A legal team ingests contracts, invoices, and policies into one index. A search for *"termination obligations liability"* returns policy exclusion clauses and invoice payment terms alongside contract termination clauses — equally ranked, because semantic similarity doesn't understand document categories. The agent has no reliable signal about which results to trust.

blue-data-doc-source classifies every document at ingest (`KeywordClassifier` — zero API cost) and propagates `document_type` to every chunk. A pre-filter pins the search to the right category before any KNN scoring runs.

```python
results = pipeline.search(
    "termination obligations liability",
    filters={"document_type": "contract"},  # pre-filter before KNN — only contract chunks scored
)
# → all results from contracts; policy and invoice noise eliminated
```

**Both steps in one call**

The two-step pattern can be collapsed into a single op. `DOCUMENT.FIND_THEN_SEARCH` runs the structured filter, searches text within each matched document, merges the results, and returns a single ranked list — with deduplication handled internally.

```python
# Equivalent to the two-step above — one call, same result
chunks = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {"fields.parties": {"contains": "Acme Corp"}},
    "query": "termination clause",
    "k": 5,
})

# Or in plain English via NLQueryParser:
chunks = source.execute_query_nl("find the termination clause in Acme Corp contracts")
# NLQueryParser detects FIND_THEN_SEARCH intent and builds the op dict automatically
```

**Failure 4 — Exact entity lookup returns wrong results or misses mentions**

An audit team needs every chunk that explicitly names "Acme Corp" — not the 5 most semantically similar chunks, but every occurrence, regardless of surrounding context. Vector search embeds "Acme Corp" and returns the k nearest chunks, missing exact mentions in short passages or tables where the entity name appears without rich surrounding prose to drive embedding quality.

blue-data-doc-source adds optional graph indexing. Every entity extracted by `BasicEnricher` becomes a shared node; every chunk that mentions it gets a `MENTIONS` edge. "Acme Corp" is a graph node with O(1) lookup — returns all chunks across all documents in one hop, regardless of how many there are.

```python
from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import DocumentPipeline, BasicEnricher

pipeline = DocumentPipeline(
    enricher=BasicEnricher(),
    graph_source=MemoryGraphSource("prod"),  # or NEO4JSource for persistence
)
pipeline.ingest("contract_a.pdf")
pipeline.ingest("vendor_b.pdf")
pipeline.ingest("incident_c.txt")

# O(1) entity lookup — returns every chunk that mentions "Acme Corp",
# across all three documents, with no embedding call
results = pipeline.graph_neighbors("Acme Corp")
# → [{"chunk_id": "...", "document_id": "...", "text_preview": "...", "entity_type": "PROPER_NOUN"}, ...]

# Walk the graph structurally — next chunk in reading order, section hierarchy,
# or which documents an entity directly appears in
docs_with_acme = pipeline.graph_query(
    "entity:PROPER_NOUN:acme corp",
    edge_type="APPEARS_IN",  # direct entity → document edge (one hop)
    direction="out",
)
```

The graph layer is **fully optional**. Omit `graph_source` and the pipeline is unchanged. When set, every ingest automatically populates:

| Edge | Meaning |
|---|---|
| `Document -[:HAS_SECTION]→ Section` | Document structure |
| `Section -[:HAS_CHUNK]→ Chunk` | Section membership |
| `Document -[:HAS_CHUNK]→ Chunk` | Direct chunk ownership |
| `Chunk -[:NEXT/PREV]→ Chunk` | Reading-order navigation |
| `Chunk -[:MENTIONS]→ Entity` | Entity occurrence |
| `Entity -[:APPEARS_IN]→ Document` | Direct document membership |
| `Entity -[:CO_OCCURS_WITH]→ Entity` | Co-occurrence within a chunk |

See the runnable demo: `python3 examples/feature-demos/graph_backend_demo.py`

See the runnable demos:
- `python3 examples/scenario-demos/structured_doc_query_demo.py` — all three failures end-to-end
- `python3 examples/feature-demos/document_find_demo.py` — DOCUMENT.FIND in depth
- `python3 examples/feature-demos/find_then_search_demo.py` — DOCUMENT.FIND_THEN_SEARCH in depth

**The integration story: documents alongside your other data sources**

Documents don't exist in isolation. A contract references parties that exist in your CRM. An invoice references line items that exist in your ERP. A policy references roles that exist in your HR system. General-purpose RAG frameworks can be connected to structured systems, but doing so typically requires application-specific orchestration for each combination.

`DocumentDataSource` extends blue-data's `DataSource` ABC and registers in the same `DataRegistry` as your SQL sources, graph stores, and JSON sources. Agents built on the blue-data intelligence layer can retrieve document evidence and cross-reference it with entity data from other registered sources — with a shared registration model rather than per-integration glue code.

```python
from blue_data import DataRegistry
from blue_data_doc_source import (
    DocumentDataSource, OpenAIEmbeddingModel,
    KeywordClassifier, LLMFieldExtractor,
)
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

doc_source = DocumentDataSource(
    name="legal_corpus",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=RedisVectorSource(host="localhost", port=6379),
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
)

# All sources share one registry — same registration model, source-specific query ops
registry = DataRegistry()
registry.register_source(sql_source)    # CRM: customers, vendors
registry.register_source(graph_source)  # entity relationships
registry.register_source(doc_source)    # contracts, invoices, policies

# Filter documents by extracted fields
contracts = doc_source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {"document_type": "contract", "fields.governing_law": {"contains": "California"}},
})

# Scope text retrieval to matched documents — combine with other source results in the agent
for doc in contracts:
    clauses = doc_source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "indemnification obligations",
        "filter": {"document_id": doc["document_id"]},
        "k": 3,
    })
```

**Reliable, repeatable ingestion**

Every document gets a SHA-256 content hash as its `document_id`. Re-ingesting the same file always produces the same document ID — upsert, not duplicate. Chunk IDs are also stable provided the parser, chunker, and chunk size remain unchanged between ingests.

Extracted fields persist across process restarts via `FileRecordStore`. Ingest once, the DocumentRecord survives the restart — `DOCUMENT.FIND` works immediately with no re-extraction.

```python
from blue_data_doc_source import DocumentDataSource, FileRecordStore, KeywordClassifier, LLMFieldExtractor

source = DocumentDataSource(
    name="legal_corpus",
    record_store=FileRecordStore("schemas.json"),  # atomic JSON write — survives restarts
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=redis_source,
)

# Ingest once — schema written atomically to schemas.json
source.execute_query({"op": "DOCUMENT.INGEST", "path": "contract.pdf"})

# After restart: FileRecordStore reloads schemas.json — no re-extraction needed
docs = source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {"fields.parties": {"contains": "Acme Corp"}},
})
```

## Core capabilities

### What differentiates this from LangChain / LlamaIndex / Haystack

- **blue-data DataRegistry integration** — `DocumentDataSource` extends blue-data's `DataSource` ABC; registers in the same `DataRegistry` as SQL, graph, and JSON sources using a shared registration and metadata model; each source type retains its own query vocabulary
- **Document classification and field extraction** — `KeywordClassifier` assigns a document type at ingest using keyword rules (zero API cost); `LLMFieldExtractor` extracts typed fields via one LLM call per document; `StructuredFieldExtractor` extracts against a caller-supplied Pydantic model, TypedDict, or JSON Schema with automatic prompt generation and output validation; all three work without a pre-configured classifier — pass only a `field_extractor` for generic extraction from unknown corpora; `document_type` propagates to every chunk and is filterable in KNN search
- **Structured corpus query (`DOCUMENT.FIND`)** — filter across all ingested documents by extracted field values (`{"fields.governing_law": {"contains": "California"}}`, `{"fields.total_due": {"gt": 30000}}`); currency symbols (`$€£¥`) and commas are stripped before numeric comparison; all conditions ANDed; deterministic — no embedding, no query-time LLM call
- **Graph backend — entity-centric retrieval and structural navigation** — pass any blue-data `GraphSource` as `graph_source=` to populate a full knowledge graph during ingest; nodes: `Document` (with `mime_type`, `ingested_at`, `chunk_count`), `Section`, `Chunk`, `Entity`; edges: `HAS_SECTION`, `HAS_CHUNK`, `NEXT`, `PREV`, `MENTIONS`, `APPEARS_IN`, `CO_OCCURS_WITH`; query via `DOCUMENT.GRAPH_NEIGHBORS` (O(1) exact entity lookup, no embeddings) or `DOCUMENT.GRAPH_QUERY` (general traversal from any node); fully optional — `graph_source=None` (the default) disables with zero overhead; backends: `MemoryGraphSource` (dev), `NEO4JSource` (production)
- **Persistent schema store** — `RecordStore` ABC backed by any blue-data `JSONSource`; `MemoryRecordStore` (default, in-process), `FileRecordStore` (atomic JSON write, survives restarts), `JsonSourceRecordStore(RedisJSONSource(...))` (multi-process-safe, no last-writer-wins race)
- **Content-addressed ingestion** — SHA-256 hash of file bytes is the `document_id`; re-ingesting the same file is always a safe upsert; chunk IDs are stable provided the parser, chunker, and chunk size remain unchanged

### Standard pipeline (solid foundation, not unique)

- **Multi-format parsing** — PDF (text layer + optional Tesseract OCR), plain text, and DOCX through a single interface; `PlumberPDFParser` adds structure-aware table extraction
- **Section-aware chunking** — detects markdown, numbered, ALL-CAPS, and underlined headers; keeps chunks within section boundaries; preserves section heading on every chunk
- **Full provenance** — every chunk carries `file_name`, `page_start`, `page_end`, `section_title`, and `content_type`; results are always citable
- **Rich search** — `search(rich=True)` / `DOCUMENT.SEARCH` with `"rich": true` returns ranked results with per-chunk citation, neighbor context, and `by_document` grouping; plain `search()` returns a flat list for programmatic use
- **Regex enrichment** — `BasicEnricher` extracts named entities (DATE, EMAIL, PERCENT, MONEY, PROPER_NOUN) and semantic hints with no ML dependency
- **LLM enrichment** — `LLMEnricher` extracts keyphrases, subject-predicate-object relationships, semantic flags (`is_legal`, `is_financial`, …), and language via a single LLM call per chunk; any provider via `llm_fn` injection
- **Pluggable embeddings** — `OpenAIEmbeddingModel`, `DummyEmbeddingModel`, or any custom `EmbeddingModel` subclass
- **Pluggable vector storage** — any blue-data `VectorSource`; `MemoryVectorSource` for dev, `RedisVectorSource` for production
- **Metadata pre-filtering** — scope KNN search to a document, file, section, MIME type, or page range before scoring
- **Natural language queries** — `NLQueryParser` maps plain-English instructions to `DOCUMENT.SEARCH`, `DOCUMENT.INGEST`, `DOCUMENT.DELETE`, `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.FIND`, and `DOCUMENT.FIND_THEN_SEARCH`; `parse_fn` injection for keyless use
- **Pluggable parsers** — `Parser` ABC for external processors (LlamaIndex, vision models, cloud OCR); bypass via `ingest_text` / `ingest_pages`
- **MCP server** — `platform/document-mcp` exposes all ops as tools callable from Claude Desktop

## Repository layout

```
blue-data-doc-source/
├── lib/                              # Python library (blue-data-doc-source)
│   ├── src/blue_data_doc_source/
│   └── tests/
├── platform/
│   └── document-mcp/                 # MCP server for document operations
├── docs/                             # Documentation
├── data/                             # Sample documents used by examples
├── design/                           # Architecture and design documents
└── examples/                         # Runnable examples
```

## Requirements

- Python 3.10+
- [`blue-data`](../blue-data) (local dependency)
- Runtime: `openai`, `pypdf`, `python-docx`, `redis[hiredis]` — installed automatically via `pip install`
- Optional: `[ocr]` — Tesseract OCR for scanned PDFs; also requires Tesseract on PATH (`brew install tesseract`)
- Optional: `[mcp]` — MCP server support (`platform/document-mcp`)
- Optional: OpenAI API key for `OpenAIEmbeddingModel` and `NLQueryParser`

```bash
export OPENAI_API_KEY=sk-...
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Ingest a file with no embeddings — useful for inspecting chunks before wiring up a vector store:

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("mydoc.pdf")

for r in records:
    print(r["chunk_id"], r["page_start"], r["text"][:80])
```

## End-to-end pipeline

Full pipeline: parse → chunk → enrich → embed → upsert to Redis → search.

`blue-data-doc-source` depends on `blue-data`. Pass a `VectorSource` instance (e.g. `RedisVectorSource`) from blue-data as the `vector_source`.

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

# Instantiate blue-data's RedisVectorSource
redis_source = RedisVectorSource(name="my_redis", host="localhost", port=6379)

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
    vector_source=redis_source,
    enricher=BasicEnricher(),
    chunk_size=1000,
    overlap=150,
    default_collection="my_docs",
)

# Ingest a document
records = source.execute_query({
    "op": "DOCUMENT.INGEST",
    "path": "contract.pdf",
    "collection": "contracts",
})
print(f"Ingested {len(records)} chunks")

# Search
results = source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "k": 5,
})
for r in results:
    print(r["chunk_id"], r["text"][:120])
```

### Without enrichment or a vector source

Embeddings and enrichment are both optional. Omit either to skip that step:

```python
source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),  # embeds but no storage
)
records = source.execute_query({"op": "DOCUMENT.INGEST", "path": "report.txt"})
```

Upsert only fires when both `embedding_model` and `vector_source` are provided and the chunk has an embedding.

## CLI example

```bash
# Ingest the sample policy — no embeddings, just inspect chunks
python examples/backends/ingest_txt.py

# Ingest with dummy embeddings
python examples/backends/ingest_txt.py --embedding dummy

# Ingest your own file
python examples/backends/ingest_txt.py path/to/doc.txt --embedding dummy --query "eligibility requirements"
```

## Local end-to-end demo (PDF → Redis → search)

`examples/integration-demos/pdf_redis_demo.py` runs the full pipeline against a local Redis Stack instance.
Sample data is included in `data/` — no file needed to get started.

### Prerequisites

**Redis Stack** (provides RediSearch + RedisJSON):

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

> If Redis is already running on port 6379 (e.g. from a previous session), this command will fail — that's fine, just skip it and proceed.

**Install dependencies** (includes `redis[hiredis]`):

```bash
pip install -e .
```

### Run

```bash
# Zero-config — uses data/sample_policy.pdf and a built-in query
python examples/integration-demos/pdf_redis_demo.py

# With OpenAI embeddings for real semantic search
OPENAI_API_KEY=sk-... python examples/integration-demos/pdf_redis_demo.py \
    --embedding openai \
    --query "what is the deductible"

# Your own PDF
OPENAI_API_KEY=sk-... python examples/integration-demos/pdf_redis_demo.py path/to/doc.pdf \
    --embedding openai \
    --query "termination clause"
```

### Natural language queries

`--nl-query` lets you ask in plain English. Under the hood, `NLQueryParser` uses an LLM (GPT-4o-mini by default) to parse the instruction into the right op and parameters, then calls `execute_query` automatically.

```python
from blue_data_doc_source import DocumentDataSource, NLQueryParser, OpenAIEmbeddingModel

source = DocumentDataSource(
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=redis_source,
    nl_parser=NLQueryParser(),
)

results = source.execute_query_nl("find the termination clause")
records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")

# DOCUMENT.FIND — structural filter, no content retrieval
docs = source.execute_query_nl("list all invoices from Blue Solutions over $30,000")

# DOCUMENT.FIND_THEN_SEARCH — filter documents, then retrieve text from them
chunks = source.execute_query_nl("find the termination clause in Acme Corp contracts")
```

You can swap in any LLM backend via `parse_fn`:

```python
parser = NLQueryParser(parse_fn=lambda text: my_llm_call(text))
```

### Options

| Flag | Default | Description |
|---|---|---|
| `path` | `data/sample_policy.pdf` | Path to a `.pdf` file |
| `--query` | built-in | Search query to run after ingestion |
| `--nl-query` | — | Natural language query — LLM parses intent automatically (requires `--embedding openai`) |
| `--embedding` | `dummy` | `dummy` (SHA-256 hash) or `openai` |
| `--openai-model` | `text-embedding-3-small` | OpenAI embedding model |
| `--chunk-size` | `300` | Max words per chunk |
| `--overlap` | `50` | Word overlap between chunks |
| `--top-k` | `3` | Number of results to return |
| `--collection` | `demo_docs` | Redis key prefix and index name |
| `--redis-host` | `localhost` | Redis host |
| `--redis-port` | `6379` | Redis port |

> **Note:** `examples/local_redis_store.py` is demo-only and is not part of the `blue_data_doc_source` package. Production use should pass Blue's `RedisVectorSource` directly as `vector_source`.

## Architecture

```
DocumentDataSource (extends blue-data DataSource)  ← public entry point
  └─ DocumentPipeline       ← orchestrates all layers; auto-detects .txt vs .pdf
       ├─ TextParser         ← reads .txt → raw string
       ├─ PDFParser          ← reads .pdf → [{page_num, text}, ...] via pypdf; OCR fallback via Tesseract
       ├─ PlumberPDFParser   ← reads .pdf → typed blocks via pdfplumber; tables as {headers, rows}
       ├─ DocxParser         ← reads .docx → markdown-formatted string via python-docx
       ├─ SectionChunker     ← detects headers, chunks within sections (default)
       │    └─ WindowChunker ← word-based sliding window fallback
       ├─ BasicEnricher      ← regex entity/keyword extraction (optional)
       ├─ KeywordClassifier  ← document type detection (no API cost) (optional)
       ├─ LLMFieldExtractor  ← typed field extraction via GPT-4o-mini (optional)
       ├─ OpenAIEmbeddingModel / DummyEmbeddingModel  (optional)
       ├─ ChunkRecord        ← canonical chunk schema with provenance fields
       ├─ VectorSource       ← blue-data VectorSource injected by caller (optional)
       └─ GraphIndexer       ← writes Document/Section/Chunk/Entity nodes + edges
            └─ GraphSource   ← blue-data GraphSource (MemoryGraphSource, NEO4JSource, …) (optional)
```

### ChunkRecord schema

Every ingested chunk becomes a `ChunkRecord`, serialised via `.to_dict()`. Optional fields are omitted when `None`.

| Field | Type | Notes |
|---|---|---|
| `document_id` | str | SHA-256 content hash of the source file — stable across re-ingests |
| `chunk_id` | str | `{document_id}_chunk_{index}` |
| `source_type` | str | Always `"document"` |
| `file_name` | str | Basename of the source file |
| `mime_type` | str | `"text/plain"`, `"application/pdf"`, or `"application/vnd.openxmlformats-officedocument.wordprocessingml.document"` |
| `text` | str | Chunk text |
| `page_start` | int \| omitted | PDF only — 1-based page number |
| `page_end` | int \| omitted | PDF only — same as `page_start` in V1 |
| `section_title` | str \| omitted | Section heading detected by SectionChunker — omitted when no structure found |
| `entities` | list | Extracted entities: `{text, type}` each |
| `semantic_hints` | dict | `word_count`, `sentence_count`, `has_questions`, `has_numbers`, `keywords` |
| `document_type` | str \| omitted | Classified document type (`contract`, `invoice`, `policy`, `research_paper`, `financial_report`, `unknown`) — present when a `classifier` is set |
| `content_type` | str \| omitted | Content block type: `"text"` or `"table"` — present when using `PlumberPDFParser` or any block-aware parser |
| `table` | dict \| omitted | Structured table data `{"headers": list[str], "rows": list[list[str]]}` — present only when `content_type == "table"` |
| `embedding` | list[float] \| omitted | Present only when an embedding model is set |
| `metadata` | dict | `source_path`, `chunk_index`, `chunk_size`, `overlap`, `ingested_at`, `embedding_model`; includes `document_type` and `content_type` when set |

### Chunking

`SectionChunker` (the default) detects section headers — markdown (`# Title`), numbered (`1.2 Overview`), ALL-CAPS, and underlined (`Title\n===`) — and chunks within each section. The detected header becomes `section_title` on each chunk. When no headers are found it falls back to plain window chunking with `section_title=None`.

`WindowChunker` uses **word count** (not character count). With `chunk_size=1000, overlap=150`, each chunk is at most 1000 words and consecutive chunks share 150 words of context. PDF chunks never span page boundaries.

Pass a custom chunker to override the default:

```python
from blue_data_doc_source.source.document.chunkers import WindowChunker
pipeline = DocumentPipeline(chunker=WindowChunker(chunk_size=500, overlap=50))
```

### Enrichment

`BasicEnricher` is regex-based (no ML dependency). It extracts:
- **Entities:** `DATE`, `EMAIL`, `PERCENT`, `MONEY`, `PROPER_NOUN`
- **Semantic hints:** word/sentence count, top keywords (stopword-filtered), boolean flags

### Document classification and field extraction

Two-layer classification runs once per document at ingest time, before the chunk loop:

**Layer 1 — `KeywordClassifier`** (no API cost, no extra deps): scores the first 500 words against signal phrase sets for each document type and returns `(document_type, confidence)`.

**Layer 2 — `LLMFieldExtractor`** (optional, one GPT-4o-mini call per document): extracts typed fields — `parties`, `effective_date`, `invoice_number`, `policy_number`, `keywords`, etc. — from the document text given its classified type.

```python
from blue_data_doc_source import DocumentPipeline, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),  # optional; requires OPENAI_API_KEY
    embedding_model=...,
    vector_source=...,
)

records = pipeline.ingest("contract.pdf")

# document_type on every chunk
records[0]["document_type"]             # → "contract"
records[0]["metadata"]["document_type"] # → "contract"

# schema stored at ingest — no re-read needed
doc_id = records[0]["document_id"]
schema = pipeline.get_record(doc_id)
# {
#   "document_id": "doc_abc123",
#   "document_type": "contract",
#   "confidence": 0.78,
#   "classifier": "llm",
#   "fields": {"parties": ["TechCorp Inc.", "BuildRight LLC"], "governing_law": "Delaware", ...}
# }
```

Via `DocumentDataSource`:

```python
schema = source.execute_query({
    "op": "DOCUMENT.GET_RECORD",
    "document_id": doc_id,
})
```

Supported types: `contract`, `invoice`, `policy`, `research_paper`, `financial_report`, `unknown`.

See the runnable demo: `python3 examples/feature-demos/field_extraction_demo.py`

### Embeddings

| Class | Use case |
|---|---|
| `OpenAIEmbeddingModel` | Production; calls `text-embedding-3-small` by default (1536-dim) |
| `DummyEmbeddingModel` | Tests; SHA-256 hash → fixed-dim float vector (not real similarity) |

All embedding models expose a `dim` property used by `ensure_index` and validated against the actual embedding length on every chunk. `OpenAIEmbeddingModel` supports `text-embedding-3-small` (1536), `text-embedding-3-large` (3072), and `text-embedding-ada-002` (1536).

### Pluggable parsers

The built-in parsers (`TextParser`, `PDFParser`) cover standard digitally-created files. To use an external document processor — LlamaIndex, a vision model, a cloud OCR API — implement the `Parser` ABC and pass it as `parser=` to `DocumentPipeline`. Everything downstream (chunking, enrichment, embedding, storage) is unchanged.

```python
from blue_data_doc_source import DocumentPipeline, Parser

class LlamaIndexParser(Parser):
    def parse(self, path: str):
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return [{"text": doc.text} for doc in docs]

pipeline = DocumentPipeline(parser=LlamaIndexParser(), embedding_model=...)
pipeline.ingest("complex_layout.pdf")  # LlamaIndex parses; our pipeline chunks and stores
```

`parse()` must return a list of dicts with `"text": str`. Include `"page_num": int` (1-based) for page-aware documents — it propagates to `page_start` / `page_end` on each `ChunkRecord`.

#### Bypassing the parser entirely

When you already have extracted text or pages from an external tool, skip file parsing altogether:

```python
# Pre-extracted text (no file path needed)
pipeline.ingest_text(text, file_name="report.txt")

# Pre-extracted pages from a vision model
pipeline.ingest_pages(
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}],
    file_name="contract.pdf",
)
```

`document_id` is derived from a SHA-256 hash of the content, so calling these methods twice with the same text is safe — it overwrites rather than duplicates.

### Scanned PDF support (OCR)

By default, `PDFParser` uses pypdf to read the text layer embedded in a PDF. Pages with no text layer — scanned documents, photographed pages — are silently skipped.

Pass `ocr_fallback=True` to run Tesseract OCR on any blank page instead of dropping it:

```python
from blue_data_doc_source.source.document.parsers import PDFParser

parser = PDFParser(ocr_fallback=True)
pages = parser.parse_pages("scanned_contract.pdf")
```

OCR'd pages carry `"ocr": True` in the result dict so downstream code can identify them.

**Install OCR dependencies:**

```bash
pip install blue-data-doc-source[ocr]   # installs pytesseract + pdf2image
brew install tesseract                   # macOS — Tesseract must be on PATH
```

> `pdf2image` also requires [Poppler](https://poppler.freedesktop.org/). On macOS: `brew install poppler`. On Ubuntu: `apt-get install poppler-utils`.

A clear `ImportError` is raised if OCR is requested but the dependencies are not installed.

### Table extraction (PlumberPDFParser)

`PDFParser` extracts the raw text layer — table cells become flat prose with no row/column structure. For agentic reasoning over tabular data, use `PlumberPDFParser` instead. It uses `pdfplumber` to detect table boundaries, extract headers and rows as structured data, and return typed blocks.

```python
from blue_data_doc_source import DocumentPipeline, PlumberPDFParser

pipeline = DocumentPipeline(
    parser=PlumberPDFParser(),
    embedding_model=...,
    vector_source=...,
)
records = pipeline.ingest("invoice.pdf")

# Table chunks carry structured data
for r in records:
    if r.get("content_type") == "table":
        print(r["table"]["headers"])  # ["Description", "Qty", "Unit Price", "Total"]
        print(r["table"]["rows"])     # [["Widget", "5", "$10", "$50"], ...]
        print(r["text"])              # pipe-delimited markdown — for LLM consumption

# Filter search to table chunks only
results = pipeline.search("unit price", filters={"content_type": "table"})
```

`content_type` is a filterable TAG field — scope KNN search to `"text"` or `"table"` chunks before scoring. Requires `pip install pdfplumber`.

### Vector source and index creation

`DocumentDataSource` and `DocumentPipeline` accept any `VectorSource` from blue-data (e.g. `RedisVectorSource`, `MemoryVectorSource`). The pipeline calls `.upsert()` on ingest and `.knn()` on search — no adapter layer needed.

Before the first ingest, create the vector index using `DOCUMENT.ENSURE_INDEX`:

```python
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})
```

Or directly on the pipeline:

```python
pipeline.ensure_index(collection="contracts")
```

This calls `VectorSource.create_index` with the correct embedding dimension. If the index already exists the behaviour depends on the backend (Redis raises an error; call it once at startup).

## Running tests

```bash
# Unit tests (no external services required)
.venv/bin/pytest lib/tests/unit/ -v

# Integration tests (requires Redis Stack on localhost:6379)
docker run -p 6379:6379 redis/redis-stack-server:latest
.venv/bin/pytest lib/tests/integration/ -v
```
