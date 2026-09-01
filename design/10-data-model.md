# blue-data-doc-source — Data Model

## Overview

`blue-data-doc-source` processes documents through a strict one-way pipeline. Each stage produces a typed data structure that is passed to the next stage. This document describes every data structure in the pipeline, its fields, and its invariants.

The pipeline stages and their output types:

```
Parser          → list[page_dict]
DocumentClassifier → (document_type, confidence)
FieldExtractor  → dict[field_name, value]     → DocumentRecord
Chunker         → list[(section_title, text)]
Enricher        → {entities, semantic_hints}
EmbeddingModel  → list[float]
ChunkRecord     → dict via to_dict()
GraphIndexer    → (side effect: nodes + edges)
RecordStore     → (side effect: DocumentRecord persisted)
```

---

## 1. Parser output — page dict

Produced by `Parser.parse(path)`. A list of page dicts, one per page or logical section.

```python
# Minimum contract
{"text": str}

# Full with optional fields
{"text": str, "page_num": int}   # page_num is 1-based
```

`parse_blocks()` extends this with a `"type"` field:

```python
# Text block
{"type": "text", "text": str, "page_num"?: int}

# Table block (PlumberPDFParser)
{
    "type": "table",
    "text": str,                             # markdown-rendered table text
    "table": {
        "headers": list[str],
        "rows": list[list[str]],
    },
    "page_num"?: int,
}
```

**Invariant:** `Parser.parse()` must always return a list. An unreadable or empty document returns `[]`. `DocumentPipeline` returns `[]` chunks when the parser returns no pages.

---

## 2. DocumentRecord

Produced by `DocumentClassifier` + `FieldExtractor` and persisted in `RecordStore`. Returned by `DOCUMENT.GET_RECORD` and `DOCUMENT.FIND`.

```python
{
    "document_id":   str,    # SHA-256 hash of file bytes; canonical document identity
    "document_type": str,    # e.g. "contract", "invoice", "policy", "research", "unknown"
    "confidence":    float,  # classification confidence in [0, 1]; 0.0 when unknown
    "classifier":    str,    # "keyword" | "llm" | name of the classifier used
    "fields":        dict,   # type-specific extracted fields; keys depend on document_type
}
```

**`fields` examples by document type:**

| `document_type` | Typical field keys |
|----------------|--------------------|
| `contract` | `parties`, `governing_law`, `effective_date`, `termination_date` |
| `invoice` | `vendor`, `invoice_number`, `invoice_date`, `total_due`, `line_items` |
| `policy` | `effective_date`, `covered_items`, `exclusions` |
| `research` | `authors`, `abstract`, `keywords`, `doi` |

**Invariant:** `FieldExtractor.extract()` omits fields it cannot determine rather than setting them to `None`. `fields` is always a dict but may be empty (`{}`).

**Storage key:** `document_id` is used as the store key in all `RecordStore` implementations.

---

## 3. Chunker output — (section_title, text) tuple

Produced by `Chunker.chunk_with_titles(text)`. A list of pairs.

```python
list[tuple[Optional[str], str]]
# (section_title, chunk_text)
# section_title is None when the chunk has no named section
```

**Invariants:**
- `SectionChunker` returns `section_title=None` for content before the first heading and for fallback window chunks when no headings are detected.
- `WindowChunker` always returns `section_title=None`.
- `SentenceChunker` always returns `section_title=None`.
- Chunk text is never empty — `DocumentPipeline` skips chunks where `text.strip()` is falsy.

---

## 4. Enricher output

Produced by `Enricher.enrich(text)`. Always a dict with exactly two keys.

```python
{
    "entities": list[{"text": str, "type": str}],
    "semantic_hints": dict[str, Any],
}
```

### BasicEnricher entity types

| Type | Detection |
|------|-----------|
| `PROPER_NOUN` | Regex: two or more consecutive title-case words (`[A-Z][a-z]+`) |

### BasicEnricher semantic_hints

| Hint key | Value | Condition |
|----------|-------|-----------|
| `has_numeric` | `True` | Chunk contains digit sequences |
| `has_date` | `True` | Chunk contains date-like strings |
| `is_short` | `True` | Chunk is below a word-count threshold |

### LLMEnricher

Returns the same structure but `entities` includes typed entities (PERSON, ORG, LOCATION, DATE, etc.) and `semantic_hints` includes `keyphrases`, `relationships`, and semantic flags. One GPT-4o-mini call per chunk.

**Invariant:** `Enricher.enrich()` always returns both keys. Empty lists/dicts are valid.

---

## 5. ChunkRecord

The canonical output unit of `DocumentPipeline`. One record per chunk per document. Assembled after all pipeline stages complete. Serialised via `to_dict()`, which omits `None` optional fields.

```python
@dataclass
class ChunkRecord:
    # Required — always present in to_dict() output
    document_id:   str
    chunk_id:      str
    text:          str
    file_name:     str
    source_type:   str = "document"
    mime_type:     str = "text/plain"
    entities:      list[dict]       # from Enricher.enrich()
    semantic_hints: dict            # from Enricher.enrich()
    metadata:      dict             # pipeline-level metadata (ingested_at, etc.)

    # Optional — omitted from to_dict() when None
    page_start:    Optional[int]    # 1-based; set by PDF parsers
    page_end:      Optional[int]    # 1-based; equal to page_start for PDF (chunks don't span pages)
    section_title: Optional[str]    # from SectionChunker; None for window/sentence chunkers
    embedding:     Optional[list[float]]  # from EmbeddingModel; omitted by to_dict()
    document_type: Optional[str]    # from DocumentClassifier
    content_type:  Optional[str]    # "text" | "table"; from parse_blocks()
    table:         Optional[dict]   # {"headers": [...], "rows": [...]}; for table chunks
```

**ID fields:**

| Field | Format | Notes |
|-------|--------|-------|
| `document_id` | `sha256(file_bytes).hexdigest()` | Deterministic; same file always produces same ID; idempotent re-ingest |
| `chunk_id` | `{document_id}_chunk_{index:04d}` | 0-based index within the document |

**Invariants:**

- `chunk_id` is deterministic — the same file always produces the same chunk IDs. Re-ingesting a file is an upsert at the vector store, never a duplicate.
- `page_start == page_end` for PDF chunks in the current implementation — chunks are scoped to a single page and never span page boundaries.
- `to_dict()` omits `None` fields. Callers must use `record.get("field")`, not `record["field"]`, for optional fields.
- When `document_type` is set, `to_dict()` also writes it into `metadata["document_type"]` for Redis Search tag filtering.
- When `content_type` is set, `to_dict()` also writes it into `metadata["content_type"]`.
- `embedding` is omitted by `to_dict()` — it is passed separately to `VectorSource.upsert()` inside `DocumentPipeline`.

---

## 6. Graph node and edge schema

Written by `GraphIndexer` during `DOCUMENT.INGEST` when `graph_source` is set. All node IDs are deterministic.

### Node types

| Node label | ID format | Properties |
|-----------|-----------|-----------|
| `Document` | `doc:{document_id}` | `document_id`, `file_name`, `document_type`, `mime_type`, `ingested_at`, `chunk_count` |
| `Section` | `section:{document_id}:{title.lower()}` | `section_id`, `title`, `document_id`, `position` |
| `Chunk` | `chunk:{chunk_id}` | `chunk_id`, `document_id`, `text_preview`, `chunk_index`, `page`?, `section_title`? |
| `Entity` | `entity:{entity_type}:{name.lower()}` | `name`, `entity_type` |

### Edge types

| Edge | Source → Target | Notes |
|------|----------------|-------|
| `HAS_SECTION` | `Document → Section` | Structural containment |
| `HAS_CHUNK` | `Section → Chunk` | When chunk belongs to a named section |
| `HAS_CHUNK` | `Document → Chunk` | Direct edge, regardless of section |
| `NEXT` | `Chunk → Chunk` | Sequential forward link |
| `PREV` | `Chunk → Chunk` | Sequential backward link |
| `MENTIONS` | `Chunk → Entity` | Entity appears in chunk |
| `CO_OCCURS_WITH` | `Entity → Entity` | Both entities appear in the same chunk |
| `APPEARS_IN` | `Entity → Document` | Entity appears anywhere in document |

**Invariants:**

- `Entity` nodes are shared across documents — the same entity name+type always maps to the same node ID. They are never removed by `DOCUMENT.DELETE`.
- `Document`, `Section`, and `Chunk` nodes are removed on `DOCUMENT.DELETE`.
- `GraphIndexer` is a no-op when `graph_source=None` — no overhead or error.
- Re-ingesting the same file skips the `Document` node creation (node already exists) and updates `Section` / `Chunk` nodes.

---

## 7. EmbeddingModel contract

```python
class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_text(self, text: str) -> List[float]: ...
```

`DocumentPipeline` calls `model.dim` to configure the vector index on `DOCUMENT.ENSURE_INDEX`. The actual embedding length is validated against `dim` on the first chunk of every ingest — a mismatch raises immediately.

---

## 8. RecordStore record

The dict structure stored and returned by `RecordStore`:

```python
{
    "document_id":   str,    # storage key
    "document_type": str,
    "confidence":    float,
    "classifier":    str,
    "fields":        dict,
}
```

`RecordStore.find(filter)` supports dot notation for nested field access and operators `contains`, `lt`, `gt`, `lte`, `gte`. Currency symbols (`$€£¥`) and commas are stripped before numeric comparison.

---

## Data Flow Summary

```
File path
  → Parser.parse()                  → list[page_dict]
  → DocumentClassifier.classify()   → (document_type, confidence)
  → FieldExtractor.extract()        → fields dict
  → RecordStore.upsert()            (side effect: DocumentRecord persisted)
  → Chunker.chunk_with_titles()     → list[(section_title, text)]
  → Enricher.enrich()               → {entities, semantic_hints}
  → EmbeddingModel.embed_text()     → list[float]
  → ChunkRecord(...)
  → to_dict() + VectorSource.upsert() (side effect)
  → GraphIndexer.index()            (side effect, if graph_source set)

DocumentPipeline.ingest() returns list[ChunkRecord.to_dict()]
DocumentDataSource.execute_query() returns that list directly to the caller
```
