# blue-data-doc-source — Library

---

## DocumentPipeline

Orchestrates the full ingestion chain: parse → chunk → enrich → embed → store. All optional components (`embedding_model`, `vector_source`, `enricher`) can be omitted; upsert fires only when both `embedding_model` and `vector_source` are set.

### Constructor

```python
DocumentPipeline(
    chunk_size: int = 1000,
    overlap: int = 150,
    chunker: Chunker | None = None,
    parser: Parser | None = None,
    embedding_model: EmbeddingModel | None = None,
    vector_source: VectorSource | None = None,
    enricher: Enricher | None = None,
    collection: str = "documents",
    ocr_fallback: bool = False,
    classifier: DocumentClassifier | None = None,
    field_extractor: FieldExtractor | None = None,
    record_store: RecordStore | None = None,
    graph_source: GraphSource | None = None,
    graph_database: str = "documents",
)
```

- `chunker` — defaults to `SectionChunker(chunk_size, overlap)` when omitted
- `parser` — when provided, overrides built-in extension detection; all extensions are routed through the custom parser
- `ocr_fallback` — passes `ocr_fallback=True` to the built-in `PDFParser`; has no effect when a custom `parser` is set
- `classifier` — when set, classifies the document type once per ingest (before the chunk loop) and propagates `document_type` to every `ChunkRecord`; defaults to `None` (no classification)
- `field_extractor` — when set alongside `classifier`, extracts typed fields from the document text via a single call per ingest; result stored in `record_store`; defaults to `None`
- `record_store` — backend for schema records; defaults to `MemoryRecordStore()` (in-process); pass `FileRecordStore(path)` for persistence across restarts
- `graph_source` — any blue-data `GraphSource` (`MemoryGraphSource`, `NEO4JSource`, etc.); when set, `GraphIndexer` writes Document/Section/Chunk/Entity nodes and edges during every ingest; `None` (the default) disables the graph layer with zero overhead
- `graph_database` — graph database name used by `GraphIndexer`; defaults to `"documents"`

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `ingest(path, collection=None)` | `list[dict]` | Parse, chunk, enrich, and embed a `.txt`, `.pdf`, or `.docx` file; upsert when `embedding_model` and `vector_source` are set |
| `ingest_text(text, file_name, mime_type, collection=None)` | `list[dict]` | Bypass file parsing — chunk, enrich, and embed pre-extracted text |
| `ingest_pages(pages, file_name, mime_type, collection=None)` | `list[dict]` | Bypass file parsing — chunk, enrich, and embed pre-extracted page dicts (`{"page_num": int, "text": str}`) |
| `search(query, k=5, collection=None, filters=None, mode="vector", rich=False, context_chunks=1)` | `list[dict]` or `dict` | Embed query and run KNN; `rich=False` (default) returns flat list; `rich=True` returns structured dict with provenance, neighbor context, and `by_document` grouping |
| `ensure_index(collection=None)` | `None` | Create vector index for the collection; idempotent — safe to call multiple times |
| `delete(path=None, document_id=None, chunk_count=None, collection=None)` | `int` | Delete all chunks for a document; returns count of successful delete calls |
| `get_record(document_id)` | `dict \| None` | Return the schema record stored during ingest; `None` if no `classifier` was set or the document is unknown |
| `find_records(filter=None)` | `list[dict]` | Return all schema records matching the filter dict; see `RecordStore.find()` for filter syntax; returns `[]` when no classifier is set |
| `graph_neighbors(entity_name, entity_type=None, k=10)` | `list[dict]` | Exact entity lookup via graph (deterministic node ID, no corpus scan, no embeddings); requires `graph_source` |
| `graph_query(node_id, edge_type=None, direction="both", k=None)` | `list[dict]` | General graph traversal from any node ID; requires `graph_source` |

### Document identity and deduplication

`document_id` is `"doc_" + sha256(file_bytes)[:16]`. Chunk IDs follow the pattern `{document_id}_chunk_{index:06d}`. Re-ingesting an unchanged file produces identical IDs — upsert overwrites in place. A changed file triggers deletion of the previous chunks (same-process only) before the new ones are written.

---

## DocumentDataSource

Top-level public entry point. Extends blue-data's `DataSource` ABC and wraps a single `DocumentPipeline` instance. Dispatches on the `"op"` field of the request dict passed to `execute_query`.

### Constructor

```python
DocumentDataSource(
    name: str = "document",
    embedding_model: EmbeddingModel | None = None,
    vector_source: VectorSource | None = None,
    enricher: Enricher | None = None,
    chunk_size: int = 1000,
    overlap: int = 150,
    default_collection: str = "documents",
    nl_parser: NLQueryParser | None = None,
    properties: dict | None = None,
    ocr_fallback: bool = False,
    classifier: DocumentClassifier | None = None,
    field_extractor: FieldExtractor | None = None,
    record_store: RecordStore | None = None,
    graph_source: GraphSource | None = None,
    graph_database: str = "documents",
)
```

### Ops

| Op | Required fields | Optional fields | Returns |
|----|-----------------|-----------------|---------|
| `DOCUMENT.INGEST` | `path` | `collection` | `list[dict]` — one dict per chunk |
| `DOCUMENT.SEARCH` | `query` | `collection`, `k` (default 5), `filter` (dict), `mode`, `rich` (bool, default false), `context_chunks` (int, default 1) | `list[dict]` — ranked chunks; or `dict` with `results`, `by_document`, provenance, and neighbor context when `rich=true` |
| `DOCUMENT.ENSURE_INDEX` | — | `collection` | `None` |
| `DOCUMENT.DELETE` | `path` OR (`document_id` + `chunk_count`) | `collection` | `int` — deleted chunk count |
| `DOCUMENT.GET_RECORD` | `document_id` | — | `dict` — schema record; raises if unknown |
| `DOCUMENT.FIND` | — | `filter` (dict) | `list[dict]` — schema records matching filter; see `RecordStore.find()` for filter syntax |
| `DOCUMENT.FIND_THEN_SEARCH` | `query` | `find_filter` (dict), `k` (default 5), `collection`, `mode` (`"vector"` \| `"bm25"` \| `"hybrid"` \| `"rerank"`) | `list[dict]` — merged ranked chunks from all documents matching `find_filter`; returns `[]` when filter matches nothing |
| `DOCUMENT.GRAPH_NEIGHBORS` | `entity` | `entity_type`, `k` (default 10) | `list[dict]` — chunks mentioning the named entity; requires `graph_source` |
| `DOCUMENT.GRAPH_QUERY` | `node_id` | `edge_type`, `direction` (default `"both"`), `k` | `list[dict]` — neighbouring nodes with `node_id`, `label`, `properties`, `edge_type`, `direction`; requires `graph_source` |

### Methods

| Method | Description |
|--------|-------------|
| `execute_query(query, database=None, collection=None, optional_properties=None)` | Dispatch on `query["op"]` |
| `execute_query_nl(text, ...)` | Parse a plain-English instruction via `NLQueryParser` and execute; raises if no `nl_parser` was provided |

---

## Parser (ABC)

Extracts raw text from a source file. Each implementation returns a list of page dicts — `{"text": str}` at minimum, with `"page_num": int` for page-aware formats.

```python
class Parser(ABC):
    def parse(self, path: str) -> list[dict]: ...
    def parse_blocks(self, path: str) -> list[dict]: ...
    # Default parse_blocks() wraps parse() adding type="text" to every block.
    # Override to return typed blocks: {"type": "text"|"table", "text": str,
    #   "page_num"?: int, "table"?: {"headers": list[str], "rows": list[list[str]]}}
```

### Built-in parsers

| Class | Input | Notes |
|-------|-------|-------|
| `TextParser` | `.txt` | Single page dict; no page_num |
| `PDFParser` | `.pdf` | One dict per page; `page_num` set; OCR fallback via Tesseract |
| `DocxParser` | `.docx` | Single page dict; headings converted to markdown `#` prefixes; tables to pipe-delimited rows |
| `PlumberPDFParser` | `.pdf` | Typed blocks via pdfplumber; text outside tables → `type="text"`; each table → `type="table"` with structured `table` dict; `parse()` flattens to page dicts for backward compat |

`DocumentPipeline.ingest()` auto-detects which parser to use from the file extension. Pass `parser=` to override.

See [parsers.md](../docs/parsers.md) for full reference and custom parser implementation.

---

## Chunker (ABC)

Splits text into overlapping `(section_title, chunk_text)` pairs. All chunkers measure size in **words**.

```python
class Chunker(ABC):
    def chunk_with_titles(self, text: str) -> list[tuple[str | None, str]]: ...
```

### Built-in chunkers

| Class | Behaviour |
|-------|-----------|
| `SectionChunker` | Detects markdown/numbered/ALL-CAPS/underlined headers; chunks within sections; falls back to window chunking when no headers found. Default. |
| `WindowChunker` | Word-count sliding window; no section awareness; `section_title=None` always |
| `SentenceChunker` | Closes chunks at sentence boundaries; requires `[nlp]` extra for best results |

See [chunkers.md](../docs/chunkers.md) for tuning guidance and custom chunker implementation.

---

## EmbeddingModel (ABC)

Converts text to a fixed-length float vector. The `dim` property is used by `ensure_index` and validated against every embedding returned by `embed_text`.

```python
class EmbeddingModel(ABC):
    @property
    def dim(self) -> int: ...
    def embed_text(self, text: str) -> list[float]: ...
```

### Built-in models

| Class | dim | Notes |
|-------|-----|-------|
| `OpenAIEmbeddingModel` | 1536 (default) | Calls `text-embedding-3-small`; supports `text-embedding-3-large` (3072) and `ada-002` (1536); requires `OPENAI_API_KEY` |
| `DummyEmbeddingModel` | configurable | SHA-256 hash → fixed-dim float vector; deterministic; not real similarity; for tests only |

See [extending.md](../docs/extending.md) for custom `EmbeddingModel` implementation.

---

## Enricher (ABC)

Extracts structured metadata from a chunk of text. Called once per chunk during ingestion.

```python
class Enricher(ABC):
    def enrich(self, text: str) -> dict: ...
    # Returns: {"entities": list[{"text": str, "type": str}], "semantic_hints": dict}
```

### Built-in enricher

**`BasicEnricher`** — regex-based, no ML dependency.

Entities extracted:

| Type | Pattern |
|------|---------|
| `DATE` | ISO dates, US dates, month-name dates |
| `EMAIL` | RFC-style email addresses |
| `PERCENT` | Numeric percentage values |
| `MONEY` | Currency symbols + amounts; dollar/euro/pound words |
| `PROPER_NOUN` | Sequences of title-case words |

Semantic hints computed:

| Key | Type | Description |
|-----|------|-------------|
| `word_count` | int | Total words in the chunk |
| `sentence_count` | int | Sentences (split on `.!?`) |
| `has_questions` | bool | Whether the chunk contains `?` |
| `has_numbers` | bool | Whether the chunk contains any digit sequence |
| `keywords` | list[str] | Top-N non-stopword words by frequency (default N=10) |

**`LLMEnricher`** — one LLM call per chunk via an injected `llm_fn: (str) -> str`. Provider-agnostic.

```python
LLMEnricher(
    llm_fn: Callable[[str], str],
    fallback: Enricher | None = None,  # used when LLM call or JSON parse fails
    max_chunk_chars: int = 2000,       # truncates chunk text before sending to LLM
)
```

Fields extracted per chunk:

| Field | Location | Description |
|-------|----------|-------------|
| `entities` | top-level list | `[{text, type}]`; types: `PERSON`, `ORG`, `LOCATION`, `DATE`, `MONEY`, `PERCENT`, `PRODUCT`, `LEGAL_TERM`, `REGULATION` |
| `keyphrases` | `semantic_hints` | List of key phrases extracted from the chunk |
| `relationships` | `semantic_hints` | Subject-predicate-object triples: `[{subject, predicate, object}]` |
| `semantic_flags` | `semantic_hints` | Boolean flags: `is_legal`, `is_financial`, `is_policy`, `is_technical`, `is_hr`, `contains_obligation`, `contains_risk` |
| `language` | `semantic_hints` | ISO 639-1 language code (e.g. `"en"`) |
| `summary` | `semantic_hints` | One-sentence summary of the chunk |
| `word_count` | `semantic_hints` | Computed locally — not from LLM |
| `sentence_count` | `semantic_hints` | Computed locally — not from LLM |

On LLM call failure or JSON parse error, falls back to `fallback.enrich(text)` if provided, otherwise returns a minimal dict with computed `word_count` / `sentence_count` only.

See [extending.md](../docs/extending.md) for custom `Enricher` implementation and combiner pattern.

---

## ChunkRecord

Canonical chunk schema. Constructed by `DocumentPipeline._make_record` and serialised via `.to_dict()`. Optional fields are omitted from the dict when `None`.

| Field | Type | Notes |
|-------|------|-------|
| `document_id` | str | `"doc_" + sha256(file_bytes)[:16]` — stable across re-ingests |
| `chunk_id` | str | `{document_id}_chunk_{index:06d}` |
| `source_type` | str | Always `"document"` |
| `file_name` | str | Basename of the source file |
| `mime_type` | str | `text/plain`, `application/pdf`, or DOCX MIME type |
| `text` | str | Chunk text |
| `page_start` | int \| omitted | 1-based page number (PDF only) |
| `page_end` | int \| omitted | Same as `page_start` in V1 |
| `section_title` | str \| omitted | Section heading from `SectionChunker`; omitted when no structure detected |
| `document_type` | str \| omitted | Classified type; built-in values are `contract`, `invoice`, `policy`, `research_paper`, `financial_report`, `unknown`; user-defined types (via `KeywordClassifier(extra_types=...)`) are also valid; present when `classifier` is set |
| `content_type` | str \| omitted | Content block type: `"text"` or `"table"`; present when using `PlumberPDFParser` or any block-aware parser |
| `table` | dict \| omitted | `{"headers": list[str], "rows": list[list[str]]}`; present only when `content_type == "table"` |
| `entities` | list | `[{"text": str, "type": str}, ...]` from enricher; empty list when no enricher |
| `semantic_hints` | dict | Scalar metadata from enricher; empty dict when no enricher |
| `embedding` | list[float] \| omitted | Present only when an embedding model is set |
| `metadata` | dict | `source_path`, `chunk_index`, `chunk_size`, `overlap`, `ingested_at`, `embedding_model`; OCR pages add `ocr: True` and `ocr_confidence`; includes `document_type` when a classifier is set |

---

## NLQueryParser

Translates plain-English instructions into `DOCUMENT.*` op dicts, then hands off to `execute_query`. Used by `DocumentDataSource.execute_query_nl`.

### Constructor

```python
NLQueryParser(
    model: str = "gpt-4o-mini",
    parse_fn: Callable[[str], dict] | None = None,
)
```

When `parse_fn` is provided it overrides the built-in OpenAI call entirely. The callable receives the raw instruction text and must return a valid op dict.

### Example

```python
from blue_data_doc_source import NLQueryParser

# Default: GPT-4o-mini
parser = NLQueryParser()

# Custom backend
parser = NLQueryParser(parse_fn=lambda text: my_llm_call(text))
```

The parsed op dict is passed directly to `DocumentDataSource.execute_query`, so the return type and error behaviour are identical to a direct op call.

---

## DocumentClassifier (ABC)

Identifies the document type from its text content. Called once per document during ingest, before the chunk loop.

```python
class DocumentClassifier(ABC):
    def classify(self, text: str) -> tuple[str, float]: ...
    # Returns: (document_type, confidence)  where confidence ∈ [0, 1]
    # Return ("unknown", 0.0) when no type can be determined.
```

### Built-in classifier

**`KeywordClassifier`** — no API cost, no extra dependencies.

Scores the first `sample_words` words (default 500) of the document against signal phrase sets for each type. Confidence is the fraction of signal phrases matched. Returns `"unknown"` when confidence falls below `min_confidence` (default 0.15).

```python
KeywordClassifier(
    min_confidence: float = 0.15,
    sample_words: int = 500,
    extra_types: dict[str, list[str]] | None = None,
)
```

- `extra_types` — maps a custom type name to a list of signal phrases. Merged with the built-in type signals at init time; user-defined entries take precedence on name collision. No subclassing required.

```python
# Example: add "lease" as a recognised type
KeywordClassifier(
    extra_types={
        "lease": ["lease agreement", "landlord", "tenant", "monthly rent", "security deposit"],
    }
)
```

Supported built-in document types and representative signals:

| Type | Signal phrases (sample) |
|------|------------------------|
| `contract` | agreement, parties, governing law, termination, indemnification, warranties |
| `invoice` | invoice, due date, amount due, bill to, line item, purchase order |
| `policy` | policy, eligibility, coverage, effective date, deductible, premium |
| `research_paper` | abstract, methodology, references, conclusion, hypothesis, doi |
| `financial_report` | revenue, net income, balance sheet, earnings per share, fiscal year |

---

## FieldExtractor (ABC)

Extracts typed fields from a document given its classified type. Called once per document during ingest when a `field_extractor` is set. A `classifier` is not required — when no classifier is present the pipeline passes `document_type="unknown"` to `extract()`.

```python
class FieldExtractor(ABC):
    def extract(self, text: str, document_type: str) -> dict: ...
    # Returns a plain dict of type-specific field names → values.
    # Omit fields that cannot be determined; do not set them to None.
```

### Built-in extractor

**`LLMFieldExtractor`** — one GPT-4o-mini call per document. Requires `openai>=1.0` and `OPENAI_API_KEY`.

```python
LLMFieldExtractor(
    model: str = "gpt-4o-mini",
    sample_chars: int = 4000,   # truncates input sent to the LLM
    extra_types: dict[str, str] | None = None,
)
```

- `extra_types` — maps a custom type name to a field-spec string describing what to extract. Merged with the built-in field specs at init time; user-defined entries take precedence on name collision. When the document type is unknown and no spec exists, the extractor falls back to a generic "extract key facts" prompt.

```python
# Example: add extraction spec for a "lease" type
LLMFieldExtractor(
    extra_types={
        "lease": (
            "Extract: monthly_rent (number), lease_term_months (int), "
            "landlord (string), tenant (string), security_deposit (number)"
        ),
    }
)
```

Builds a type-specific prompt from the document text and returns a JSON dict of extracted fields. Falls back to `{"raw_response": "..."}` if the LLM returns malformed JSON.

**`StructuredFieldExtractor`** — extraction driven by a caller-supplied schema. No predefined field list required. Requires `openai>=1.0` and `OPENAI_API_KEY`.

```python
StructuredFieldExtractor(
    schema: type | dict,         # Pydantic model class, TypedDict, or JSON Schema dict
    model: str = "gpt-4o-mini",
    sample_chars: int = 4000,
)
```

The extraction prompt is generated automatically from the schema definition. For Pydantic models the LLM output is validated via `model_validate()` and type-coerced before storage; falls back to the raw parsed dict on validation error. TypedDict and JSON Schema dict inputs return the parsed dict as-is.

`document_type` is ignored — `StructuredFieldExtractor` always extracts against the supplied schema regardless of how the document was classified.

```python
from pydantic import BaseModel

class LeaseAgreement(BaseModel):
    tenant: str
    monthly_rent: float
    lease_term_months: int

pipeline = DocumentPipeline(
    field_extractor=StructuredFieldExtractor(schema=LeaseAgreement),
)
```

### Schema record format

The schema stored in `pipeline.record_store` (and returned by `get_record` / `DOCUMENT.GET_RECORD`) has the following shape:

```python
{
    "document_id":   "doc_abc123",
    "chunk_id":      "doc_abc123_schema",   # keyed but not upserted to vector store
    "document_type": "contract",
    "confidence":    0.78,
    "classifier":    "keyword",             # "llm" when field_extractor is set
    "fields":        {                      # {} when no field_extractor
        "parties":       ["TechCorp Inc.", "BuildRight Solutions LLC"],
        "governing_law": "Delaware",
        "effective_date": "September 1, 2026",
    },
}
```

Schema persistence is controlled by the injected `RecordStore`. The default `MemoryRecordStore` is in-process only. Pass `FileRecordStore(path)` to persist across restarts — the file is written atomically on every upsert.

---

## RecordStore (ABC)

Pluggable backend for per-document schema records. Injected into `DocumentPipeline` and `DocumentDataSource` via `record_store=`. Defaults to `MemoryRecordStore()` when omitted.

```python
class RecordStore(ABC):
    def upsert(self, record: dict) -> None: ...
    def get(self, document_id: str) -> dict | None: ...
    def find(self, filter: dict | None = None) -> list[dict]: ...
    def delete(self, document_id: str) -> None: ...
```

### Built-in implementations

**`MemoryRecordStore`** — in-process dict; default when no `record_store=` is passed.

**`FileRecordStore(path)`** — JSON file backend; reads on init, writes atomically on every `upsert` and `delete` (write-to-tmp then `os.replace`). Survives process restarts. Creates the file on first write; handles empty or missing files safely.

### `find()` filter syntax

```python
# Exact match (case-insensitive for strings)
store.find({"document_type": "contract"})

# Dot notation for nested fields
store.find({"fields.governing_law": "State of Delaware"})

# Substring match
store.find({"fields.vendor": {"contains": "Blue Solutions"}})

# Numeric comparison — strips $€£¥ and commas before comparing
store.find({"fields.total_due": {"gt": 30000}})
store.find({"fields.contract_value": {"lte": 50000}})

# ISO date string lexicographic comparison
store.find({"fields.effective_date": {"lt": "2026-01-01"}})

# Multiple conditions — all ANDed
store.find({
    "document_type": "invoice",
    "fields.vendor": {"contains": "Acme"},
    "fields.total_due": {"gt": 10000},
})
```

Supported operators: `contains`, `gt`, `lt`, `gte`, `lte`. Pass `None` or `{}` to return all records.
