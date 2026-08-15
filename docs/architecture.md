# Architecture

blue-data-doc-source is a strict one-way pipeline. Data flows in one direction only — from file to vector store — with no shared state between layers and no backwards coupling.

```
File (.txt / .pdf / .docx)
  │
  ▼
Parser              extract raw text or page-by-page text
  │
  ▼
Chunker             split into overlapping word-windows, section-aware
  │
  ▼
Enricher            regex entity + keyword extraction (optional)
  │
  ▼
EmbeddingModel      embed each chunk text (optional)
  │
  ▼
ChunkRecord         canonical schema — document_id, chunk_id, provenance fields
  │
  ▼
VectorSource        upsert to Redis or any blue-data VectorSource (optional)
```

Each layer is independently pluggable. You can replace any stage without touching the others.

---

## Layers

### DocumentDataSource

`DocumentDataSource` extends blue-data's `DataSource` ABC. It is the public entry point for the library.

Responsibilities:
- Accepts `execute_query({"op": "...", ...})` calls
- Dispatches on the `"op"` field to `DOCUMENT.INGEST`, `DOCUMENT.SEARCH`, `DOCUMENT.ENSURE_INDEX`, `DOCUMENT.DELETE`
- Owns a single `DocumentPipeline` instance reused across all calls
- Exposes `execute_query_nl(text)` for natural language dispatch via `NLQueryParser`

`DocumentDataSource` adds no logic of its own beyond dispatch. All ingestion, search, and deletion logic lives in `DocumentPipeline`.

### DocumentPipeline

`DocumentPipeline` orchestrates the ingestion chain. It is the core of the library.

Responsibilities:
- Auto-detects `.txt`, `.pdf`, `.docx` by file extension; raises `ValueError` for unsupported types
- Delegates to the injected `Parser`, or uses the built-in `TextParser` / `PDFParser` / `DocxParser`
- Calls the `Chunker` on each page/document text
- Calls the `Enricher` on each chunk (if set)
- Calls `EmbeddingModel.embed_text()` on each chunk (if set)
- Constructs a `ChunkRecord` per chunk and calls `VectorSource.upsert()` (if both embedding and vector source are set)
- Maintains an in-process `_doc_registry` for stale-chunk cleanup when a file is re-ingested with changed content

Key methods:

| Method | Description |
|---|---|
| `ingest(path, collection)` | Full pipeline from file |
| `ingest_text(text, file_name)` | Skip parsing — feed pre-extracted text |
| `ingest_pages(pages, file_name)` | Skip parsing — feed pre-extracted page dicts |
| `search(query, k, collection)` | Embed query → KNN via VectorSource |
| `ensure_index(collection)` | Create vector index (idempotent) |
| `delete(path=...)` | Remove all chunks from the vector store |

### Parser

`Parser` is an ABC with two methods:

```python
class Parser(ABC):
    def parse(self, path: str) -> List[Dict[str, Any]]: ...
    def parse_blocks(self, path: str) -> List[Dict[str, Any]]: ...
```

`parse()` returns a list of page dicts — `{"text": str}` at minimum, with `"page_num": int` (1-based) for page-aware formats. `parse_blocks()` returns typed content blocks: each dict carries `"type": "text"|"table"`, `"text"`, and optionally `"page_num"` and `"table": {"headers": list[str], "rows": list[list[str]]}`. The default `parse_blocks()` wraps `parse()` adding `type="text"` — override it to return real table blocks.

Built-in implementations:

| Class | Input | Notes |
|---|---|---|
| `TextParser` | `.txt` | Single page dict; no `page_num` |
| `PDFParser` | `.pdf` | One dict per page; `page_num` set; OCR fallback via Tesseract |
| `DocxParser` | `.docx` | Single page dict; headings → markdown `#` prefixes; tables → pipe-delimited rows |
| `PlumberPDFParser` | `.pdf` | Typed blocks via pdfplumber; text outside tables → `type="text"`; each table → `type="table"` with structured `table` dict; `parse()` flattens to page dicts for backward compat |

`DocxParser` converts heading styles to markdown (`# Heading 1`, `## Heading 2`, ...) and tables to pipe-delimited rows, so `SectionChunker` can detect structure from Word documents the same way it does from plain markdown.

When `parser=` is passed to `DocumentPipeline`, the built-in parsers are bypassed entirely and the injected parser handles all file types.

### Chunker

`Chunker` is an ABC with a single method:

```python
class Chunker(ABC):
    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        ...
```

Returns `(section_title, chunk_text)` pairs. `section_title` is `None` when the chunk has no detected parent section.

Built-in implementations:

| Class | Strategy |
|---|---|
| `SectionChunker` | Detects headers → chunks within each section (default) |
| `WindowChunker` | Word-count sliding window, no section awareness |
| `SentenceChunker` | Closes chunks at sentence boundaries, never mid-sentence |

`SectionChunker` is the default. It uses `WindowChunker` internally to chunk within each detected section. When no headers are found it delegates entirely to `WindowChunker` and returns `section_title=None` on all chunks.

### Enricher

`Enricher` is an ABC:

```python
class Enricher(ABC):
    def enrich(self, text: str) -> Dict[str, Any]:
        # must return {"entities": [...], "semantic_hints": {...}}
```

`BasicEnricher` is the only built-in implementation. It uses regular expressions — no ML dependency. Extracted entities carry `{"text": str, "type": str}`. Entity types: `DATE`, `EMAIL`, `PERCENT`, `MONEY`, `PROPER_NOUN`.

Semantic hints always include: `word_count`, `sentence_count`, `has_questions`, `has_numbers`, `keywords` (top-k non-stopword terms by frequency).

When no `Enricher` is set, `entities` and `semantic_hints` on each `ChunkRecord` are left empty.

### EmbeddingModel

`EmbeddingModel` is an ABC:

```python
class EmbeddingModel(ABC):
    @property
    def dim(self) -> int: ...
    def embed_text(self, text: str) -> List[float]: ...
```

The `dim` property is required and must match the length of the vector returned by `embed_text`. `DocumentPipeline` validates this on every chunk and raises `ValueError` on mismatch.

`dim` is also passed to `VectorSource.create_index` during `ensure_index`.

| Class | dim | Notes |
|---|---|---|
| `OpenAIEmbeddingModel` | 1536 / 3072 / 1536 | `text-embedding-3-small` (default), `text-embedding-3-large`, `text-embedding-ada-002` |
| `DummyEmbeddingModel` | configurable | SHA-256 hash → float vector; deterministic; not real similarity |

### ChunkRecord

`ChunkRecord` is a dataclass — the canonical unit of output from the pipeline.

```python
@dataclass
class ChunkRecord:
    document_id: str        # "doc_{sha256[:16]}" of file bytes
    chunk_id: str           # "{document_id}_chunk_{index:06d}"
    text: str
    file_name: str
    source_type: str = "document"
    mime_type: str = "text/plain"
    page_start: Optional[int] = None   # PDF/page-aware only
    page_end: Optional[int] = None     # same as page_start in V1
    section_title: Optional[str] = None
    content_type: Optional[str] = None # "text" or "table" (block-aware parsers only)
    table: Optional[Dict] = None       # {"headers": [...], "rows": [[...]]} when content_type == "table"
    entities: List[Dict] = ...
    semantic_hints: Dict = ...
    metadata: Dict = ...
    embedding: Optional[List[float]] = None
```

`to_dict()` serialises the record and **omits** `None` optional fields — `page_start`, `page_end`, `section_title`, `content_type`, `table`, and `embedding` are absent from the dict when `None`, not serialised as `null`. Downstream consumers should check field presence.

`content_type` and `table` are set when the pipeline uses a block-aware parser such as `PlumberPDFParser`. Table blocks bypass the chunker — one table becomes one chunk.

`metadata` always contains: `source_path`, `chunk_index`, `chunk_size`, `overlap`, `ingested_at`, `embedding_model`. OCR pages additionally carry `"ocr": True` and `"ocr_confidence": float`. When `content_type` is set, it is also mirrored into `metadata["content_type"]` for vector-store filtering.

### VectorSource

`VectorSource` is a blue-data abstraction. `DocumentPipeline` accepts any implementation that provides:

- `upsert(collection, doc_id, doc)` — called on every embedded chunk
- `knn(index_name, query_vector, k)` — called on search
- `create_index(index_name, collection, dim)` — called by `ensure_index`
- `delete(collection, doc_id)` — called by `delete` and stale-chunk cleanup

No adapter layer is needed. Pass any `VectorSource` instance directly as `vector_source=`.

---

## Deduplication and stale-chunk cleanup

`document_id` is derived from the SHA-256 hash of the file's bytes. Re-ingesting the same file produces the same `document_id` and the same `chunk_id` values — upsert overwrites rather than duplicates.

When a file is modified and re-ingested, the content hash changes and a new `document_id` is generated. The pipeline detects this via the in-process `_doc_registry` and deletes the old chunks before upserting the new ones.

The registry is in-process only. Cross-process stale-chunk cleanup requires an explicit `DOCUMENT.DELETE` call with `document_id` and `chunk_count` from the previous ingest response.

---

## Extension points

| To change... | Inject via... |
|---|---|
| Document parser | `DocumentPipeline(parser=MyParser())` |
| Chunking strategy | `DocumentPipeline(chunker=SentenceChunker())` |
| Enrichment logic | `DocumentPipeline(enricher=MyEnricher())` |
| Embedding model | `DocumentPipeline(embedding_model=MyModel())` |
| Vector store backend | `DocumentPipeline(vector_source=MyVectorSource())` |
| NL query parsing | `DocumentDataSource(nl_parser=NLQueryParser(parse_fn=...))` |

Pre-extracted text or pages can be fed directly to `pipeline.ingest_text()` or `pipeline.ingest_pages()`, bypassing file I/O entirely.
