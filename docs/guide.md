# Developer Guide

## Quick start

Parse and chunk a file with no embeddings — useful for inspecting chunk structure before wiring up a vector store:

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("policy.pdf")

for r in records:
    print(r["chunk_id"], r.get("section_title"), r["text"][:80])
```

`DocumentPipeline` auto-detects the parser by extension: `.txt` → `TextParser`, `.pdf` → `PDFParser`, `.docx` → `DocxParser`.

---

## Full pipeline

The complete ingestion chain: parse → chunk → enrich → embed → upsert to vector store → search.

```python
from blue_data_doc_source import DocumentPipeline, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

pipeline = DocumentPipeline(
    chunk_size=500,
    overlap=100,
    enricher=BasicEnricher(),
    embedding_model=OpenAIEmbeddingModel(),   # requires OPENAI_API_KEY
    vector_source=MemoryVectorSource(),
    collection="contracts",
)

# Create the vector index before the first ingest
pipeline.ensure_index(collection="contracts")

# Ingest documents
pipeline.ingest("contract_a.pdf")
pipeline.ingest("contract_b.pdf")

# Search
results = pipeline.search("termination clause", k=5)
for r in results:
    print(r["chunk_id"], r["text"][:120])
```

Upsert fires only when both `embedding_model` and `vector_source` are provided and the chunk has an embedding. Omit either to skip persistence and just return records.

---

## DocumentDataSource

`DocumentDataSource` wraps `DocumentPipeline` behind blue-data's `DataSource` interface, exposing ops as a dict-based `execute_query` call. Use this when integrating with `DataRegistry` or the MCP server.

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    enricher=BasicEnricher(),
    chunk_size=500,
    overlap=100,
    default_collection="contracts",
)

# Create index
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})

# Ingest
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
```

### Supported ops

| Op | Required fields | Returns |
|---|---|---|
| `DOCUMENT.INGEST` | `path` | `list[dict]` — one dict per chunk |
| `DOCUMENT.SEARCH` | `query` | `list[dict]` — ranked chunks; add `"rich": true` for provenance + context + `by_document` grouping |
| `DOCUMENT.ENSURE_INDEX` | — | `None` |
| `DOCUMENT.DELETE` | `path` OR (`document_id` + `chunk_count`) | `int` — deleted chunk count |
| `DOCUMENT.GET_RECORD` | `document_id` | `dict` — DocumentRecord; raises if document unknown or no classifier was set |
| `DOCUMENT.FIND` | — | `list[dict]` — DocumentRecords matching `filter`; no embedding, no LLM, deterministic |
| `DOCUMENT.FIND_THEN_SEARCH` | `query` | `list[dict]` — filter documents by `find_filter`, search text within matched documents, return ranked chunks |

---

## Registering with DataRegistry

`DocumentDataSource` extends blue-data's `DataSource` ABC, so it registers in the same `DataRegistry` as SQL, graph, and JSON sources. An agent or MCP client talking to the registry does not need to know that documents use `DOCUMENT.SEARCH` while a SQL source uses `SELECT` — each source's `NLQueryParser` handles the translation internally.

### Register and ingest

```python
from blue_data.registry.data_registry import DataRegistry
from blue_data.source.json.memory_json_source import MemoryJSONSource
from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser

registry = DataRegistry(
    store=MemoryJSONSource(),           # registry metadata store
    vector_source=MemoryVectorSource(), # registry search engine
    embedding_fn=my_embed_fn,           # for registry metadata search
)

doc_source = DocumentDataSource(
    name="legal_corpus",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),  # doc chunk store (separate from registry store)
    classifier=KeywordClassifier(),
    nl_parser=NLQueryParser(),           # satisfies the DataSource ABC's NL contract
    default_collection="legal_corpus",
)
registry.register_source(doc_source)

# Ingest via the registry
registry.execute_query(
    source="legal_corpus",
    query={"op": "DOCUMENT.INGEST", "path": "contract.pdf", "collection": "legal_corpus"},
)
```

### sync_source — schema discovery

After ingesting, call `sync_source` to have the registry walk the live source and record its schema (databases → collections → ChunkRecord attributes). This is what powers registry-level discoverability and metadata search.

```python
registry.sync_source("legal_corpus", recursive=True)

# Registry now knows the ChunkRecord schema
databases   = registry.list_source_databases("legal_corpus")
collections = registry.get_source_database_collections("legal_corpus", "documents")
entities    = registry.get_source_database_collection_entities(
    "legal_corpus", "documents", "legal_corpus"
)
attrs = registry.get_source_database_collection_entity_attributes(
    "legal_corpus", "documents", "legal_corpus", "ChunkRecord"
)
print([a["name"] for a in attrs])
# ['document_id', 'chunk_id', 'text', 'file_name', 'mime_type',
#  'page_start', 'page_end', 'section_title', 'document_type', ...]
```

`sync_source` is non-destructive — run it any time after new documents are ingested. It adds new entries and removes stale ones without touching existing records.

### search_records — registry metadata search

Search across all registered sources by their schema descriptions:

```python
# Find which registered source holds document chunks
hits = registry.search_records("document chunks", page_size=5)
for h in hits:
    print(h["name"], h["type"], h["scope"])
```

This searches registry metadata (source names, entity descriptions, attribute descriptions), not the document content. Use `DOCUMENT.SEARCH` for content retrieval.

### execute_query and execute_query_nl

```python
# Structured op via registry
contracts = registry.execute_query(
    source="legal_corpus",
    query={"op": "DOCUMENT.FIND", "filter": {"document_type": "contract"}},
)

chunks = registry.execute_query(
    source="legal_corpus",
    query={"op": "DOCUMENT.SEARCH", "query": "termination clause", "k": 5},
)

# Natural language — dispatched through NLQueryParser
result = registry.execute_query_nl(
    source="legal_corpus",
    text="find the indemnification clause in Acme Corp contracts",
)
```

`execute_query_nl` requires `nl_parser=NLQueryParser()` on the `DocumentDataSource`. Without it, the method raises `ValueError`.

See `examples/integration-demos/data_registry_demo.py` for a fully runnable demo (no API key required).

---

## Chunking

The default chunker is `SectionChunker`. It detects section headers (markdown, numbered, ALL-CAPS, underlined) and chunks within each section, preserving the heading as `section_title` on every chunk. Falls back to word-count windowing when no headers are found.

```python
# Default: SectionChunker
pipeline = DocumentPipeline(chunk_size=500, overlap=100)

# Override with WindowChunker for unstructured prose
from blue_data_doc_source import WindowChunker
pipeline = DocumentPipeline(chunker=WindowChunker(chunk_size=300, overlap=50))

# SentenceChunker — no sentence ever split across chunks (requires [nlp] extra for best results)
from blue_data_doc_source import SentenceChunker
pipeline = DocumentPipeline(chunker=SentenceChunker(chunk_size=400, overlap=60))
```

All chunkers measure size in **words**, not characters or tokens. See [chunkers.md](chunkers.md) for tuning guidance and custom chunker implementation.

---

## Enrichment

`BasicEnricher` is regex-based — no ML dependencies. It extracts named entities and computes semantic hints, both stored on each `ChunkRecord`.

```python
from blue_data_doc_source import BasicEnricher

pipeline = DocumentPipeline(enricher=BasicEnricher(), chunk_size=500, overlap=100)
records = pipeline.ingest("policy.pdf")

for r in records:
    print(r["entities"])        # [{"text": "January 15, 2026", "type": "DATE"}, ...]
    print(r["semantic_hints"])  # {"word_count": 312, "keywords": [...], ...}
```

`LLMEnricher` makes one LLM call per chunk and extracts richer metadata: extended entity types (`PERSON`, `ORG`, `LOCATION`, `LEGAL_TERM`, `REGULATION`, …), keyphrases, subject-predicate-object relationships, semantic flags (`is_legal`, `is_financial`, …), language, and a summary. Provider-agnostic via `llm_fn: (str) -> str`.

```python
from blue_data_doc_source import LLMEnricher, BasicEnricher

enricher = LLMEnricher(
    llm_fn=lambda text: my_llm_call(text),
    fallback=BasicEnricher(),  # used on LLM failure or parse error
)
pipeline = DocumentPipeline(enricher=enricher, ...)
```

Enrichment is optional — omit `enricher` and `entities` / `semantic_hints` are empty on every record. See [extending.md](extending.md) to implement a custom enricher.

---

## Embeddings

| Class | Use | Notes |
|---|---|---|
| `OpenAIEmbeddingModel` | Production | Requires `OPENAI_API_KEY`; default model `text-embedding-3-small` (1536-dim) |
| `DummyEmbeddingModel` | Tests | SHA-256 hash → fixed-dim float vector; not real similarity |

```python
from blue_data_doc_source import OpenAIEmbeddingModel, DummyEmbeddingModel

# Production
model = OpenAIEmbeddingModel(model="text-embedding-3-small")  # 1536-dim

# Testing (no API key needed)
model = DummyEmbeddingModel(dim=8)
```

All embedding models expose a `dim` property used by `ensure_index` and validated against the actual embedding length on every chunk. See [extending.md](extending.md) to implement a custom embedding model.

---

## Vector storage

Pass any `VectorSource` from blue-data as `vector_source`:

```python
# In-memory — development and testing
from blue_data.source.vector.memory_vector_source import MemoryVectorSource
vs = MemoryVectorSource()

# Redis Stack — production (requires redis-stack-server)
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
vs = RedisVectorSource(name="docs", host="localhost", port=6379)
```

Always call `ensure_index` (or `DOCUMENT.ENSURE_INDEX`) before the first ingest. The pipeline does not auto-create the index.

---

## Search

`DocumentPipeline.search` embeds the query and runs KNN against the vector store:

```python
results = pipeline.search("payment obligations", k=5, collection="contracts")
for r in results:
    print(r["chunk_id"], r.get("section_title"), r["text"][:120])
```

`DocumentDataSource.execute_query` with `DOCUMENT.SEARCH` delegates to the same path.

### Pre-filtering

Pass a `filters` dict to scope the KNN search before it runs. Pre-filtering is applied at the vector store level — only the matching subset of chunks is scored.

```python
# Scope to a single document
results = pipeline.search(
    "termination clause",
    filters={"document_id": "doc_3a7f1c2e"},
)

# Scope to a file type
results = pipeline.search(
    "revenue figures",
    filters={"mime_type": "application/pdf"},
)

# Scope to a specific file
results = pipeline.search(
    "eligibility",
    filters={"file_name": "policy_2024.pdf"},
)

# Scope to a section
results = pipeline.search(
    "notice period",
    filters={"section_title": "Termination"},
)

# Page range (PDFs only — chunks without page_start are excluded)
results = pipeline.search(
    "definitions",
    filters={"page_start": {"gte": 1, "lte": 10}},
)

# Exact page
results = pipeline.search("liability cap", filters={"page_start": 5})

# Combine filters (AND)
results = pipeline.search(
    "payment terms",
    filters={"file_name": "contract_a.pdf", "section_title": "Pricing"},
)
```

Via the op interface:

```python
source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "filter": {"document_id": "doc_3a7f1c2e"},
})
```

**Supported filter keys:**

| Key | Type | Match |
|-----|------|-------|
| `document_id` | str | Exact — scope to one document |
| `file_name` | str | Exact — scope to one file |
| `section_title` | str | Exact — scope to one section heading |
| `mime_type` | str | Exact — scope to a file type |
| `content_type` | str | Exact — scope to `"text"` or `"table"` chunks |
| `page_start` | int or `{"gte": N, "lte": M}` | Numeric range — PDF chunks only |

All filters are combined with AND. `None` values are ignored. Chunks where an optional field is absent (e.g. `page_start` on a TXT chunk) are excluded when that field is filtered.

**Index requirement:** pre-filtering requires an index created after this feature was added. Existing Redis indexes without the filterable fields must be dropped and recreated with `ensure_index()`.

---

## Rich search

`search_rich()` returns an enriched response in a single call — ranked chunks with per-chunk provenance, neighbor context, and results grouped by document.

```python
response = pipeline.search_rich("termination clause", k=5, context_chunks=1)

response["query"]           # original query string
response["result_count"]    # number of matched chunks
response["document_count"]  # number of distinct documents
response["results"]         # ranked list of enriched result dicts
response["by_document"]     # results grouped by document_id
```

Each result carries:
- `rank`, `score`, `chunk_id`, `text`
- `provenance` — `{document_id, file_name, chunk_id, chunk_index}` plus `page` (PDF) and `section` (when detected)
- `document_type` — present when a classifier was set at ingest
- `context` — `{before, after}` with neighboring chunk text; keys omitted when no neighbor exists

```python
for r in response["results"]:
    print(r["rank"], r["provenance"]["file_name"], r["provenance"].get("page"))
    print(r["text"][:120])
    if "context" in r:
        print("  before:", r["context"].get("before", "")[:60])
        print("  after: ", r["context"].get("after",  "")[:60])
```

`by_document` groups matches per source document:

```python
for doc_id, group in response["by_document"].items():
    print(group["file_name"], group["match_count"])
    if "top_sections" in group:
        print("  sections:", group["top_sections"])
```

Via `DocumentDataSource`:

```python
response = source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause",
    "collection": "contracts",
    "k": 5,
    "rich": True,
    "context_chunks": 1,          # neighbors before/after each result
    "filter": {"document_id": "doc_abc123"},
})
```

Set `context_chunks=0` to omit neighbor context entirely. Neighbor lookup is in-process only — chunks ingested in a previous process run are not available as neighbors.

---

## Table extraction (PlumberPDFParser)

`PDFParser` extracts the raw text layer — table cells become flat prose with no row/column structure, which reduces LLM reasoning accuracy on tabular data. Use `PlumberPDFParser` when documents contain tables that agents need to query.

```python
from blue_data_doc_source import DocumentPipeline, PlumberPDFParser

pipeline = DocumentPipeline(
    parser=PlumberPDFParser(),
    embedding_model=...,
    vector_source=...,
)
records = pipeline.ingest("invoice.pdf")
```

`PlumberPDFParser` uses pdfplumber to detect table bounding boxes on each page, then returns typed content blocks:

- Text outside tables → `content_type="text"` chunks, processed by the chunker normally
- Each table → `content_type="table"` chunk (atomic, bypasses the chunker)

Table chunks carry both forms of the table:
- `text` — pipe-delimited markdown (for LLM consumption, consistent with all other chunks)
- `table` — structured dict (for machine-readable use)

```python
for r in records:
    if r.get("content_type") == "table":
        headers = r["table"]["headers"]   # ["Description", "Qty", "Unit Price", "Total"]
        rows    = r["table"]["rows"]      # [["Widget", "5", "$10.00", "$50.00"], ...]
        print(r["text"])                  # | Description | Qty | Unit Price | Total |
                                          # | ---         | --- | ---        | ---   |
                                          # | Widget      | 5   | $10.00     | $50.00|
```

`content_type` is indexed as a filterable TAG — scope search to table chunks only:

```python
results = pipeline.search("unit price", filters={"content_type": "table"})
```

**Install:** `pip install pdfplumber`

### Parser ABC — parse_blocks()

`PlumberPDFParser` extends the `Parser` ABC's new `parse_blocks()` method, which returns a flat list of typed block dicts:

```python
blocks = PlumberPDFParser().parse_blocks("invoice.pdf")
# [
#   {"type": "text",  "page_num": 1, "text": "Invoice header..."},
#   {"type": "table", "page_num": 1, "text": "| Desc | ... |", "table": {...}},
#   {"type": "text",  "page_num": 1, "text": "Payment terms..."},
# ]
```

All `Parser` subclasses inherit a default `parse_blocks()` that wraps `parse()` and marks every block `type="text"`. Override it to add block-aware extraction.

---

## Document classification and field extraction

`KeywordClassifier` and `LLMFieldExtractor` run once per document at ingest time to identify its type and extract typed fields. `document_type` is then propagated to every chunk produced from that document.

### Layer 1 — KeywordClassifier (no API cost)

Scores the first 500 words against signal phrase sets for five document types. Zero extra dependencies.

```python
from blue_data_doc_source import DocumentPipeline, KeywordClassifier

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    embedding_model=...,
    vector_source=...,
)

records = pipeline.ingest("contract.pdf")

records[0]["document_type"]             # → "contract"
records[0]["metadata"]["document_type"] # → "contract"  (same value in metadata)
```

Supported types: `contract`, `invoice`, `policy`, `research_paper`, `financial_report`, `unknown`.

Confidence threshold defaults to `0.15` — below this the type is `"unknown"`. Tune it:

```python
KeywordClassifier(min_confidence=0.25)   # stricter — more unknowns
KeywordClassifier(sample_words=200)      # faster — score fewer words
```

**Adding custom document types** — pass `extra_types` with a mapping of type name to signal phrase list. Merged with the built-in types; user-supplied entries take precedence on name collision. Built-in types are unchanged.

```python
classifier = KeywordClassifier(
    extra_types={
        "lease": [
            "lessor", "lessee", "rent", "premises", "landlord",
            "tenant", "lease term", "security deposit",
        ],
        "employment": [
            "employer", "employee", "salary", "at-will",
            "benefits", "job title", "probation period",
        ],
    }
)
pipeline = DocumentPipeline(classifier=classifier, ...)
records = pipeline.ingest("lease_agreement.pdf")
records[0]["document_type"]  # → "lease"
```

Custom types work end-to-end with `DOCUMENT.FIND` and pre-filters — no other changes needed:

```python
pipeline.find_records({"document_type": "lease"})
pipeline.search("security deposit", filters={"document_type": "lease"})
```

### Layer 2 — LLMFieldExtractor (optional, one call per document)

Calls GPT-4o-mini with a type-specific prompt to extract typed fields from the document. Requires `OPENAI_API_KEY`.

```python
from blue_data_doc_source import LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),  # one API call per document
    ...
)
```

Fields extracted per type:

| Type | Fields |
|------|--------|
| `contract` | `parties`, `effective_date`, `expiration_date`, `governing_law`, `contract_value`, `key_obligations` |
| `invoice` | `invoice_number`, `invoice_date`, `due_date`, `vendor`, `bill_to`, `line_items`, `subtotal`, `tax`, `total_due` |
| `policy` | `policy_number`, `effective_date`, `expiration_date`, `insured`, `coverage_types`, `deductible`, `premium`, `exclusions` |
| `research_paper` | `title`, `authors`, `abstract`, `keywords`, `methodology`, `findings`, `doi` |
| `financial_report` | `company`, `fiscal_period`, `revenue`, `net_income`, `earnings_per_share`, `total_assets`, `total_liabilities`, `highlights` |

**Adding fields for custom types** — pair `extra_types` on `LLMFieldExtractor` with the same type name used in `KeywordClassifier`. The value is a plain-English extraction instruction in the same format as the built-in specs.

```python
extractor = LLMFieldExtractor(
    extra_types={
        "lease": (
            "Extract: lessor (string), lessee (string), "
            "monthly_rent (string with currency), lease_term (string), "
            "security_deposit (string), premises_address (string)."
        ),
        "employment": (
            "Extract: employer (string), employee (string), "
            "job_title (string), start_date (ISO date string), "
            "salary (string with currency), jurisdiction (string)."
        ),
    }
)
pipeline = DocumentPipeline(
    classifier=KeywordClassifier(extra_types={"lease": [...], "employment": [...]}),
    field_extractor=extractor,
    ...
)
```

After ingest, `DOCUMENT.FIND` field filters work for custom types exactly as for built-in ones:

```python
pipeline.find_records({"document_type": "lease", "fields.lessor": {"contains": "Alice"}})
```

For document types with no matching spec (including `"unknown"`), `LLMFieldExtractor` falls back to a generic prompt: `"Extract any structured metadata you can find."`.

### Generic extraction without a classifier

Pass only `field_extractor` — no `classifier` needed. The pipeline calls `extract()` with `document_type="unknown"`, which triggers the generic fallback prompt. Useful for unknown corpora where document types are not defined in advance.

```python
pipeline = DocumentPipeline(
    field_extractor=LLMFieldExtractor(),   # no classifier
)
records = pipeline.ingest("mystery_doc.pdf")
schema = pipeline.find_records()
# → [{"document_type": "unknown", "confidence": 0.0, "fields": {...}}]
```

`find_records()` works exactly as when a classifier is set — `document_type` is `"unknown"` and field names are whatever the LLM extracted.

### StructuredFieldExtractor — schema-driven extraction

Instead of a plain-English field spec string, supply a Pydantic model, TypedDict, or JSON Schema dict. The prompt is generated automatically from the schema definition. For Pydantic models the LLM output is validated and coerced before storage.

```python
from pydantic import BaseModel
from blue_data_doc_source import StructuredFieldExtractor

class EmploymentContract(BaseModel):
    employee_name: str
    start_date: str
    base_salary: str
    notice_period: str
    non_compete: bool

pipeline = DocumentPipeline(
    field_extractor=StructuredFieldExtractor(schema=EmploymentContract),
)
pipeline.ingest("offer_letter.pdf")
schema = pipeline.find_records()[0]
# → {"document_type": "unknown", "fields": {"employee_name": "Alice", "non_compete": False, ...}}
```

Works with any of the three schema formats:

| Input | Prompt generated from | Output validation |
|---|---|---|
| Pydantic model class | `model_json_schema()` | `model_validate()` — type coercion; falls back to raw dict on error |
| TypedDict class | `__annotations__` | None — returns parsed dict as-is |
| JSON Schema dict | `properties` | None — returns parsed dict as-is |

Pair with a classifier to use structured extraction for known types:

```python
pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=StructuredFieldExtractor(schema=EmploymentContract),
)
```

`document_type` is set by the classifier; `StructuredFieldExtractor` ignores it and always extracts against the supplied schema.

### Schema retrieval

The schema is stored in-process at ingest time. Retrieve it without re-reading the document:

```python
doc_id = records[0]["document_id"]
schema = pipeline.get_record(doc_id)
```

Schema record format:

```python
{
    "document_id": "doc_abc123",
    "chunk_id":    "doc_abc123_schema",
    "document_type": "contract",
    "confidence":    0.78,
    "classifier":    "llm",       # "keyword" when no field_extractor is set
    "fields": {
        "parties":       ["TechCorp Inc.", "BuildRight Solutions LLC"],
        "governing_law": "Delaware",
        "effective_date": "September 1, 2026",
    },
}
```

Via `DocumentDataSource`:

```python
schema = source.execute_query({
    "op": "DOCUMENT.GET_RECORD",
    "document_id": "doc_abc123",
})
```

### Record persistence

By default DocumentRecords are stored in `MemoryRecordStore` — in-process only. Two durable options:

**`FileRecordStore`** — JSON file, single process. Written atomically on every ingest (`write-to-tmp` then `os.replace`) — crash-safe.

```python
from blue_data_doc_source import DocumentPipeline, FileRecordStore, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=FileRecordStore("schemas.json"),
)
# After restart, schemas.json is reloaded — DOCUMENT.FIND works immediately
```

**`JsonSourceRecordStore`** — wraps any blue-data `JSONSource` backend. Use `RedisJSONSource` for multi-process-safe durable storage — two ingest workers can write concurrently without a last-writer-wins race.

```python
from blue_data.source.json.redis.redis_json_source import RedisJSONSource
from blue_data_doc_source import DocumentPipeline, JsonSourceRecordStore, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=JsonSourceRecordStore(
        RedisJSONSource(name="doc_schemas", properties={"namespace": "doc_schemas"})
    ),
)
```

`MemoryRecordStore` and `FileRecordStore` are thin convenience wrappers over `JsonSourceRecordStore` — they use `MemoryJSONSource` and `FileJSONSource` respectively. Any blue-data `JSONSource` (including `MongoDBSource`) works as a drop-in backend.

### Structured corpus query (DOCUMENT.FIND)

`find_records()` filters across all ingested DocumentRecords — no embedding, no LLM call at query time. Records exist when either a `classifier` or a `field_extractor` (or both) is set at ingest time.

```python
# All contracts governed by California
docs = pipeline.find_records({"document_type": "contract", "fields.governing_law": {"contains": "California"}})

# Invoices over $30,000 — currency symbols stripped automatically
docs = pipeline.find_records({"document_type": "invoice", "fields.total_due": {"gt": 30000}})
```

Via `DocumentDataSource`:

```python
docs = source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {"fields.parties": {"contains": "Acme Corp"}},
})
```

**Two-step pattern** — combine with `DOCUMENT.SEARCH` for precise, noise-free retrieval:

```python
docs   = pipeline.find_records({"fields.parties": {"contains": "Acme Corp"}})
chunks = pipeline.search("termination clause", filters={"document_id": docs[0]["document_id"]})
```

#### Filter syntax reference

Filters are plain dicts. All conditions are ANDed. A missing field causes the condition to fail (the record is excluded).

**Filterable top-level fields:**

| Field | Type | Notes |
|---|---|---|
| `document_type` | string | Exact match, case-insensitive. Values: `contract`, `invoice`, `policy`, `research_paper`, `financial_report`, `unknown` |
| `confidence` | float | Confidence score from the classifier (0–1) |
| `classifier` | string | `"keyword"` or `"llm"` |

**Extracted fields** — accessed via dot notation `fields.<name>`:

```python
{"fields.governing_law": "California"}           # exact match
{"fields.vendor": {"contains": "Blue Solutions"}} # substring
{"fields.total_due": {"gt": 30000}}               # numeric comparison
{"fields.effective_date": {"gte": "2024-01-01"}}  # lexicographic (ISO dates sort correctly)
```

**Operators:**

| Operator | Behaviour |
|---|---|
| _(no operator)_ | Exact match — case-insensitive for strings |
| `contains` | Case-insensitive substring match |
| `gt`, `lt`, `gte`, `lte` | Numeric if both sides parse as numbers; lexicographic otherwise |

**Numeric coercion:** strips `$`, `€`, `£`, `¥`, and `,` before parsing. So `"$35,154"` → `35154.0`. Formats that don't parse (percentages, locale separators like `.` as thousands separator, parenthesized negatives like `(500)`) fall back to lexicographic string comparison — the filter will not error, but the result may be wrong.

**`DOCUMENT.FIND_THEN_SEARCH`** — combines a document filter with text retrieval in one call. The pipeline runs FIND internally, searches within each matched document, and returns ranked chunks. Use this when you want content from a known subset of documents:

```python
# "find the termination clause in Acme Corp contracts"
chunks = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {"document_type": "contract", "fields.parties": {"contains": "Acme Corp"}},
    "query": "termination clause",
    "k": 5,
})

# "what are the payment terms in Blue Solutions invoices over $30,000?"
chunks = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {"document_type": "invoice", "fields.vendor": {"contains": "Blue Solutions"}, "fields.total_due": {"gt": 30000}},
    "query": "payment terms",
    "k": 3,
})
```

Both `DOCUMENT.FIND` and `DOCUMENT.FIND_THEN_SEARCH` are reachable via `execute_query_nl` when an `NLQueryParser` is wired up — the parser detects whether the query is purely structural (FIND) or combines a document filter with content retrieval (FIND_THEN_SEARCH).

### Custom classifier or extractor

Implement either ABC to replace the built-in logic:

```python
from blue_data_doc_source import DocumentClassifier, FieldExtractor

class MyClassifier(DocumentClassifier):
    def classify(self, text: str) -> tuple[str, float]:
        # return (document_type, confidence)
        ...

class MyExtractor(FieldExtractor):
    def extract(self, text: str, document_type: str) -> dict:
        # return a dict of typed fields
        ...

pipeline = DocumentPipeline(
    classifier=MyClassifier(),
    field_extractor=MyExtractor(),
    ...
)
```

---

## Natural language queries

`execute_query_nl` is part of the blue-data `DataSource` ABC — every data source must implement it so a `DataRegistry` or agent can address all sources uniformly in plain English. `NLQueryParser` is doc-source's implementation: it translates a natural-language string into the appropriate `DOCUMENT.*` op dict, then hands off to `execute_query`.

Without `nl_parser=` set, `execute_query_nl` raises `ValueError` immediately — the method exists on the class because the ABC requires it, but it does nothing useful until you wire up a parser.

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=my_vector_source,
    nl_parser=NLQueryParser(),   # uses GPT-4o-mini by default; requires OPENAI_API_KEY
)

results = source.execute_query_nl("find the termination clause in the contracts collection")
# NLQueryParser maps this to: {"op": "DOCUMENT.SEARCH", "query": "termination clause", "collection": "contracts"}

records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")
# Maps to: {"op": "DOCUMENT.INGEST", "path": "/docs/policy.pdf", "collection": "contracts"}
```

Inject any `(str) -> dict` callable via `parse_fn` to use a different LLM or run without an API key:

```python
parser = NLQueryParser(parse_fn=lambda text: my_llm_call(text))
```

**NLQueryParser supported ops:** `DOCUMENT.SEARCH`, `DOCUMENT.INGEST`, `DOCUMENT.DELETE`, `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.FIND`, and `DOCUMENT.FIND_THEN_SEARCH`. `DOCUMENT.GET_RECORD` requires `execute_query` with a structured op dict — it needs an exact `document_id` which natural language cannot reliably supply.

---

## Deduplication and re-ingestion

`document_id` is derived from a SHA-256 hash of the file's bytes. Re-ingesting the same file produces the same `document_id` and the same `chunk_id` values — vector store upsert overwrites rather than duplicates.

When a file's content changes, the pipeline detects the new `document_id`, deletes the old chunks from the vector store, and upserts the new ones. This cleanup is **in-process only** — stale chunks from a previous process run must be removed explicitly:

```python
# Using path (requires same pipeline instance that did the ingest)
pipeline.delete(path="contract.pdf", collection="contracts")

# Using document_id + chunk_count (cross-process safe)
pipeline.delete(document_id="doc_abc123", chunk_count=14, collection="contracts")

# Via DocumentDataSource
source.execute_query({
    "op": "DOCUMENT.DELETE",
    "document_id": "doc_abc123",
    "chunk_count": 14,
    "collection": "contracts",
})
```

---

## Pre-extracted text and pages

Skip file parsing entirely when text has already been extracted by an external tool:

```python
# Full-document text
pipeline.ingest_text(text, file_name="report.txt", collection="reports")

# Page-level output (page_num propagates to ChunkRecord.page_start / page_end)
pipeline.ingest_pages(
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}],
    file_name="scanned.pdf",
    collection="reports",
)
```

---

## Scanned PDFs (OCR)

Enable Tesseract OCR for pages with no embedded text layer:

```python
pipeline = DocumentPipeline(ocr_fallback=True, ...)
pipeline.ingest("scanned_contract.pdf")
```

Requires the `[ocr]` extra and Tesseract on PATH. See [installation.md](installation.md) for setup instructions. OCR pages carry `metadata["ocr"] = True` and `metadata["ocr_confidence"]` on each chunk record.
