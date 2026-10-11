# Features

## Why it matters

Vector-only retrieval fails in predictable ways on a real document corpus. `blue-data-doc-source` packages the fixes as native blue-data source operations:

- **Semantic search returns the wrong document.** A query for Acme Corp's termination clause can surface unrelated contracts first (especially when one says "cancellation"). Fix: extract `parties`/`governing_law`/`effective_date` once at ingest, then [`DOCUMENT.FIND`](#structured-corpus-query-documentfind) the right document before searching within it — see [two-stage find-then-retrieve](#two-stage-find-then-retrieve-documentfind_then_search).
- **Structured questions can't be answered by embedding distance.** "Invoices from Blue Solutions over $30,000" needs a numeric predicate, not similarity. Fix: typed field extraction at ingest + deterministic `DOCUMENT.FIND` filters (currency/commas stripped before comparison).
- **A mixed corpus makes every search ambiguous.** Contracts, invoices, and policies in one index rank equally for the same query. Fix: zero-cost `KeywordClassifier` tags every chunk with `document_type`, so a pre-filter pins search to the right category before KNN scoring.
- **Exact entity lookup misses mentions.** Vector search returns the *k* nearest chunks, not *every* mention of "Acme Corp". Fix: optional [graph indexing](#graph-backend) resolves an entity by deterministic node ID and walks its `MENTIONS` edges — no corpus scan, no embeddings.

Documents also don't exist in isolation: `DocumentDataSource` registers in the same blue-data `DataRegistry` as your SQL, graph, and web sources, so an agent can cross-reference document evidence with other enterprise data through one registration model. And ingestion is reproducible — the `document_id` is a SHA-256 of file bytes, so re-ingesting is a safe upsert and extracted fields survive restarts via `FileRecordStore`.

## Multi-format document parsing

Ingest PDF, plain text, and DOCX files through a single unified interface. The parser is selected automatically by file extension.

- **TXT** — reads the full file as a single text block
- **PDF** — page-by-page extraction via pypdf; optional Tesseract OCR fallback for scanned pages
- **DOCX** — structure-preserving conversion via python-docx; headings and paragraphs mapped to markdown for section detection

```python
from blue_data_doc_source import DocumentPipeline

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
from blue_data_doc_source.source.document.chunkers import SectionChunker, WindowChunker

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

`SentenceChunker` aligns chunks to sentence boundaries — no sentence is ever split across chunk boundaries. Uses `nltk.sent_tokenize` when available; falls back to punctuation regex without NLTK.

```python
from blue_data_doc_source.source.document.chunkers import SentenceChunker

chunker = SentenceChunker(chunk_size=200, overlap=40)
pipeline = DocumentPipeline(chunker=chunker, embedding_model=..., vector_source=...)
```

Install NLTK for higher-quality sentence detection:

```bash
pip install blue-data-doc-source[nlp]
python -c "import nltk; nltk.download('punkt_tab')"
```

`SentenceChunker` produces no `section_title` — use `SectionChunker` when heading preservation matters.

Both chunkers operate on **word count**, not character count. PDF chunks never span page boundaries.

---

## Content-addressed deduplication

Every `ChunkRecord` carries a `document_id` derived from a SHA-256 hash of the source file's bytes. The same file always produces the same `document_id` and the same `chunk_id` values — re-ingesting is a safe upsert that overwrites rather than appends.

```python
from blue_data_doc_source import DocumentPipeline

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
from blue_data_doc_source.source.document.enrichment import BasicEnricher

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

## LLM enrichment

`LLMEnricher` makes a single LLM call per chunk and extracts richer metadata than regex alone: named entities with extended types, keyphrases, subject-predicate-object relationships, semantic flags, language, and a one-sentence summary.

```python
from blue_data_doc_source import LLMEnricher, BasicEnricher
import openai

def call_openai(text: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
    )
    return response.choices[0].message.content

enricher = LLMEnricher(
    llm_fn=call_openai,
    fallback=BasicEnricher(),   # used when LLM call or JSON parse fails
    max_chunk_chars=2000,       # truncate chunk before sending to LLM
)
```

Entity types extracted: `PERSON`, `ORG`, `LOCATION`, `DATE`, `MONEY`, `PERCENT`, `PRODUCT`, `LEGAL_TERM`, `REGULATION`.

Fields added to `semantic_hints`: `keyphrases`, `relationships` (subject/predicate/object triples), `semantic_flags` (`is_legal`, `is_financial`, `is_policy`, `is_technical`, `is_hr`, `contains_obligation`, `contains_risk`), `language` (ISO 639-1), `summary`. `word_count` and `sentence_count` are always computed locally regardless of LLM output.

`llm_fn` is any `(str) -> str` callable — provider-agnostic. On failure, falls back to `fallback.enrich(text)` if provided.

---

## Structured corpus query (DOCUMENT.FIND)

`DOCUMENT.FIND` filters across all ingested DocumentRecords by extracted field values — no embedding, no LLM, deterministic. Records exist when either a `classifier` or a `field_extractor` (or both) is set at ingest time.

```python
# Filter by document type
docs = pipeline.find_records({"document_type": "contract"})

# Filter by nested extracted field
docs = pipeline.find_records({"fields.governing_law": {"contains": "California"}})

# Numeric comparison — currency symbols and commas stripped automatically
docs = pipeline.find_records({"fields.total_due": {"gt": 30000}})

# Multiple conditions ANDed
docs = pipeline.find_records({
    "document_type": "invoice",
    "fields.vendor":    {"contains": "Blue Solutions"},
    "fields.total_due": {"gt": 10000},
})
```

**Filter syntax:** all conditions ANDed; missing fields exclude the record.

| Operator | Behaviour |
|---|---|
| _(no operator)_ | Exact match — case-insensitive for strings |
| `contains` | Case-insensitive substring match |
| `gt`, `lt`, `gte`, `lte` | Numeric after stripping `$€£¥` and `,`; lexicographic fallback |

Dot notation accesses nested fields: `"fields.governing_law"`, `"fields.total_due"`. Top-level filterable fields: `document_type`, `confidence`, `classifier`.

Numeric coercion handles `"$35,154"` → `35154.0`. Formats that can't be parsed (percentages, parenthesized negatives, locale separators) fall back to string comparison without erroring.

Via `DocumentDataSource`:

```python
docs = source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {"document_type": "contract", "fields.governing_law": {"contains": "Delaware"}},
})
```

DocumentRecords persist across restarts when `FileRecordStore` is used — ingest once, query forever. For multi-process deployments use `JsonSourceRecordStore(RedisJSONSource(...))` — concurrent ingest workers can write safely without a last-writer-wins race.

**Two-step agent pattern** — `DOCUMENT.FIND` narrows the corpus to the right documents, then `DOCUMENT.SEARCH` retrieves text only within those documents:

```python
# Step 1: find by structured field — deterministic
docs = pipeline.find_records({"fields.governing_law": {"contains": "California"}})

# Step 2: retrieve text scoped to matched document only
chunks = pipeline.search("termination clause", filters={"document_id": docs[0]["document_id"]})
```

> **`contains` matches string fields only.** It does a case-insensitive substring match on string-valued fields (e.g. `governing_law`, `vendor`). Matching an element of a **list-valued** field such as `fields.parties` (`["Acme Corp", …]`) is not supported yet — that filter returns no matches today (tracked in #58). Filter on a string field meanwhile.

---

## Two-stage find-then-retrieve (DOCUMENT.FIND_THEN_SEARCH)

`DOCUMENT.FIND_THEN_SEARCH` combines a document-level filter with text retrieval in a single call. It runs `DOCUMENT.FIND` internally, searches within each matched document, then returns a merged ranked list — eliminating application-level filter-then-loop logic.

```python
chunks = source.execute_query({
    "op":         "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {"document_type": "contract", "fields.governing_law": {"contains": "California"}},
    "query":      "termination clause",
    "k":          5,
})
```

The `find_filter` accepts the same syntax as `DOCUMENT.FIND`. The `query` is searched only within documents that pass the filter — not the full corpus. Results are ranked chunks, each carrying full provenance.

`NLQueryParser` detects two-part questions automatically and emits `DOCUMENT.FIND_THEN_SEARCH` when it identifies both a document type intent and a content retrieval intent in the same natural-language query.

---

## Record retrieval (DOCUMENT.GET_RECORD)

`DOCUMENT.GET_RECORD` retrieves the full `DocumentRecord` for any ingested document by `document_id`. Returns the document type, confidence score, classifier name, and all extracted fields — without touching the vector store.

```python
# Via pipeline
record = pipeline.get_record(document_id)
# {
#   "document_id": "doc_abc123",
#   "document_type": "invoice",
#   "confidence": 0.80,
#   "classifier": "llm",
#   "fields": {
#     "invoice_number": "INV-2026-004",
#     "vendor": "Acme Corp",
#     "total_due": "$1,250.00",
#   }
# }

# Via DocumentDataSource
record = source.execute_query({
    "op":         "DOCUMENT.GET_RECORD",
    "document_id": document_id,
})
```

Useful for inspecting what was extracted at ingest time, building document-level summaries, or feeding structured metadata to an agent without re-running search.

---

## Pluggable embedding models

`EmbeddingModel` is an ABC with a single `embed_text(text) -> list[float]` method and a `dim` property. Swap implementations without changing the pipeline.

```python
from blue_data_doc_source import OpenAIEmbeddingModel, DummyEmbeddingModel

# Production — calls OpenAI text-embedding-3-small (1536-dim)
openai_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
print(openai_model.dim)   # 1536

# Tests — SHA-256 hash → fixed-dim float vector (not real similarity)
dummy_model = DummyEmbeddingModel()
print(dummy_model.dim)    # 8
```

Bring your own model by subclassing `EmbeddingModel`:

```python
from blue_data_doc_source.source.document.embeddings import EmbeddingModel

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
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel

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
    vector_source=RedisVectorSource(name="redis", properties={"connection": {"host": "localhost", "port": 6379}}),
    default_collection="reports",
)

# Create index before first ingest
source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "reports"})
```

`DOCUMENT.ENSURE_INDEX` is safe to call multiple times — skip creation if the index already exists.

---

## Multi-document collections

A collection holds any number of ingested documents. `DOCUMENT.SEARCH` spans all of them in a single query. Each document is independently deletable by its `document_id` — deleting one document leaves all others in the collection untouched. Re-ingesting an updated file replaces only that document's chunks via content-hash deduplication.

```python
# Ingest multiple documents into one collection
pipeline.ingest("contracts/acme.pdf",   collection="contracts")
pipeline.ingest("contracts/beta.pdf",   collection="contracts")
pipeline.ingest("contracts/gamma.pdf",  collection="contracts")

# Search spans all three documents in one query
results = pipeline.search("termination clause", collection="contracts", k=5)

# Delete one document — others are untouched
pipeline.delete(path="contracts/beta.pdf", collection="contracts")
```

Collections are named strings — any value works. Use separate collections to partition by tenant, document category, or access scope. A pipeline instance can ingest into and search across multiple collections.

---

## Search modes

`DOCUMENT.SEARCH` supports four retrieval strategies via the `"mode"` field:

| Mode | Description | Requires embeddings |
|---|---|---|
| `"vector"` | Default KNN — cosine similarity over chunk embeddings | Yes |
| `"bm25"` | Keyword-only BM25 scoring — no embedding model needed | No |
| `"hybrid"` | Reciprocal Rank Fusion (RRF) of KNN + BM25 | Yes |
| `"rerank"` | BM25 shortlist followed by cosine re-rank | Yes |

```python
# BM25 — keyword relevance, no embedding cost
results = pipeline.search("deductible", mode="bm25", k=5)

# Hybrid — fuses semantic and keyword signals
results = pipeline.search("deductible", mode="hybrid", k=5)

# Via op dict
results = source.execute_query({
    "op":    "DOCUMENT.SEARCH",
    "query": "deductible",
    "mode":  "bm25",
    "k":     5,
})
```

`"bm25"` is useful for testing without an embedding model, for keyword-heavy corpora where semantic similarity adds noise, and for the `NLQueryParser` demo path where dummy embeddings would produce meaningless scores.

> **In-process only.** `"bm25"`, `"hybrid"`, and `"rerank"` all score against an in-memory chunk index built during `ingest()`. This index is lost when the process exits. After a restart, these modes silently return empty results even if the vector store is intact. `"vector"` mode is unaffected — it queries the persistent vector store. If you need BM25 or hybrid across restarts, re-ingest documents in the current process before searching, or use `"vector"` mode only.

---

## Natural language queries

`NLQueryParser` translates plain-English instructions into `DOCUMENT.*` op dicts using an LLM, then dispatches the resulting op automatically.

```python
from blue_data_doc_source import DocumentDataSource, NLQueryParser, OpenAIEmbeddingModel

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
from blue_data_doc_source.source.document.parsers import PDFParser

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
from blue_data_doc_source import DocumentPipeline, Parser

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

## Document classification and field extraction

Two-layer classification runs once per document at ingest time, before the chunk loop.

**Layer 1 — `KeywordClassifier`** does the type detection using no API, no ML. It takes the first 500 words of the document, lowercases them, and scores each candidate type by counting how many of its signal phrases appear in that sample. Score = `matches / total_signals`. The type with the highest score wins; if it's below `min_confidence` (default 0.15), `"unknown"` is returned.

Signal phrases per type (sample):

| Type | Signals |
|---|---|
| `contract` | agreement, parties, governing law, indemnification, warranties, breach, arbitration |
| `invoice` | invoice, due date, amount due, bill to, line item, subtotal, vendor, unit price |
| `policy` | policy, eligibility, coverage, deductible, premium, exclusion, insured, claim |
| `research_paper` | abstract, methodology, references, conclusion, hypothesis, doi, we propose |
| `financial_report` | revenue, net income, balance sheet, earnings per share, fiscal year, ebitda |

Both `KeywordClassifier` and `LLMFieldExtractor` accept an `extra_types` parameter to extend the built-in set without subclassing. Pass a dict of `{type_name: signal_phrases}` to the classifier and `{type_name: field_spec_string}` to the extractor — built-in types are unchanged and user-supplied types take precedence on name collision:

```python
KeywordClassifier(
    extra_types={"lease": ["lessor", "lessee", "rent", "premises", "landlord",
                           "tenant", "lease term", "security deposit"]}
)
LLMFieldExtractor(
    extra_types={"lease": "Extract: lessor (string), lessee (string), "
                          "monthly_rent (string with currency), lease_term (string)."}
)
```

**Layer 2 — `LLMFieldExtractor`** (optional) runs after the type is known. It sends the first 4000 characters of the document to GPT-4o-mini with a type-specific field prompt and parses the JSON response into a `fields` dict. One API call per document, not per chunk.

```python
from blue_data_doc_source import DocumentPipeline, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),  # optional — requires OPENAI_API_KEY
    embedding_model=...,
    vector_source=...,
)

records = pipeline.ingest("invoice.pdf")

# document_type is on every chunk
print(records[0]["document_type"])   # "invoice"

# full schema (type + extracted fields) via GET_RECORD
schema = pipeline.get_record(records[0]["document_id"])
# {
#   "document_id": "doc_abc123",
#   "document_type": "invoice",
#   "confidence": 0.80,
#   "classifier": "llm",
#   "fields": {
#     "invoice_number": "INV-2026-004",
#     "vendor": "Acme Corp",
#     "total_due": "$1,250.00",
#     ...
#   }
# }
```

`KeywordClassifier` alone (without `field_extractor`) costs nothing — no API key, no extra deps. You still get `document_type` on every chunk and can filter search results by type:

```python
pipeline = DocumentPipeline(classifier=KeywordClassifier(), ...)
pipeline.ingest("contract.pdf")

results = pipeline.search("termination clause", filters={"document_type": "contract"})
```

The DocumentRecord is not upserted to the vector store. Retrieve it via `pipeline.get_record(document_id)` or `DOCUMENT.GET_RECORD`. By default records are stored in `MemoryRecordStore` (in-process only). Pass `FileRecordStore(path)` to persist across restarts:

```python
from blue_data_doc_source import DocumentPipeline, FileRecordStore, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=FileRecordStore("schemas.json"),  # survives process restarts
)
```

### Generic extraction — `field_extractor` without a classifier

`LLMFieldExtractor` can be used **without** a `KeywordClassifier`. When no classifier is set, the pipeline assigns `document_type = "unknown"` and `confidence = 0.0`, then runs the field extractor on the raw document text with no type hint. The LLM discovers and names fields freely based on whatever it finds in the document.

**This mode is for schema discovery, not production filtering.** The LLM chooses field names freely and inconsistently — one document might produce `{"vendor": "Acme"}` while another produces `{"supplier_name": "Acme Corp"}` for the same concept. `DOCUMENT.FIND` filters on exact field names, so those two records would not both match `{"fields.vendor": {"contains": "Acme"}}`. Use this mode to understand what fields your corpus contains, then define a `StructuredFieldExtractor` schema once field names are understood.

```python
from blue_data_doc_source import DocumentPipeline, LLMFieldExtractor

pipeline = DocumentPipeline(
    field_extractor=LLMFieldExtractor(),   # no classifier=
)

pipeline.ingest_text(lab_notebook_text, file_name="lab_notebook.txt")
pipeline.ingest_text(incident_report_text, file_name="incident.txt")

records = pipeline.find_records()
# document_type = 'unknown', confidence = 0.0
# fields = whatever the LLM named: researcher, protocol, sample_id, ...
# Warning: field names may differ across documents for the same concept
```

Inspect what was extracted across all records before writing any filters:

```python
# See all field names the LLM chose across the corpus
for rec in pipeline.find_records():
    print(rec["document_id"], sorted(rec.get("fields", {}).keys()))
# doc_abc123: ['root_cause', 'severity', 'status']
# doc_def456: ['incident_severity', 'root_cause_analysis', 'ticket_status']
# ↑ same concepts, different names — filters on one will miss the other
```

Once you have identified which field names the LLM consistently produces, define a `StructuredFieldExtractor` schema to enforce those names going forward:

```python
from pydantic import BaseModel
from blue_data_doc_source import StructuredFieldExtractor

class IncidentReport(BaseModel):
    severity: str
    root_cause: str
    status: str

pipeline = DocumentPipeline(
    field_extractor=StructuredFieldExtractor(schema=IncidentReport),
)
# Now every document produces the same field names — filters are reliable
```


---

### Choosing your extraction setup

Start here — pick your path based on what you already know:

```
Do you need extracted fields?
  │
  ├─ No  → KeywordClassifier() alone
  │         Assigns document_type to every chunk at zero API cost.
  │         Use DOCUMENT.SEARCH with filters={"document_type": "contract"}.
  │
  └─ Yes → Do you already have a schema (Pydantic / TypedDict / JSON Schema)?
             │
             ├─ No  → LLMFieldExtractor() alone          ← schema discovery, not production filtering
             │         LLM names fields freely and inconsistently: "vendor" in one doc,
             │         "supplier_name" in another. DOCUMENT.FIND filters will miss variants.
             │         Use this to explore: find_records() → see what field names appeared →
             │         define a StructuredFieldExtractor(schema) once names are understood.
             │
             └─ Yes → Do you also need document_type on each chunk?
                        │
                        ├─ No  → StructuredFieldExtractor(schema)
                        │         Schema-defined names. Pydantic-validated output.
                        │
                        └─ Yes → KeywordClassifier() + StructuredFieldExtractor(schema)
                                  Detected document_type + schema-defined, validated fields.
```

| Scenario | Setup | `document_type` | Field names |
|---|---|---|---|
| Type classification only | `classifier=KeywordClassifier()` | detected type | — |
| Unfamiliar corpus — discover fields | `field_extractor=LLMFieldExtractor()` only | `"unknown"` | LLM-chosen, may vary per document — exploration only; define schema once names are understood |
| Known types, typed fields | `classifier=KeywordClassifier()` + `field_extractor=LLMFieldExtractor()` | detected type | type-specific prompt |
| Known schema, validated output | `field_extractor=StructuredFieldExtractor(schema)` | `"unknown"` | schema-defined, Pydantic-validated |
| Typed + schema-validated | `classifier=KeywordClassifier()` + `field_extractor=StructuredFieldExtractor(schema)` | detected type | schema-defined, Pydantic-validated |

All five paths write to the same `DocumentRecord`, use the same `DOCUMENT.FIND` filter syntax, and store fields under the same `fields.*` dot-notation. The setup decision only affects type detection accuracy and how predictable field names are.

---

## Schema-validated extraction (StructuredFieldExtractor)

`StructuredFieldExtractor` accepts a Pydantic model class, TypedDict, or JSON Schema dict and auto-generates the extraction prompt from the schema — no manual field spec string needed. For Pydantic models the output is validated and coerced via `model_validate` before storage; invalid LLM output falls back to the raw parsed dict.

```python
from pydantic import BaseModel
from blue_data_doc_source import DocumentPipeline, StructuredFieldExtractor

class EmploymentContract(BaseModel):
    employee_name: str
    start_date: str
    base_salary: str
    notice_period: str
    non_compete: bool

pipeline = DocumentPipeline(
    field_extractor=StructuredFieldExtractor(schema=EmploymentContract),
    embedding_model=...,
    vector_source=...,
)
pipeline.ingest("employment_contract.pdf")

rec = pipeline.get_record(records[0]["document_id"])
# rec["fields"] == {
#   "employee_name": "Jane Smith",
#   "start_date": "2026-01-15",
#   "base_salary": "$120,000",
#   "notice_period": "30 days",
#   "non_compete": True,
# }
```

TypedDict and JSON Schema dict also work:

```python
from typing import TypedDict

class InvoiceFields(TypedDict):
    vendor: str
    total_due: str
    due_date: str

pipeline = DocumentPipeline(
    field_extractor=StructuredFieldExtractor(schema=InvoiceFields),
)
```

`document_type` is set to `"unknown"` unless a `KeywordClassifier` is also provided — but the extraction prompt is derived entirely from the schema, so no classifier is required to get typed, predictable field names.

See `examples/feature-demos/structured_extractor_demo.py` for a runnable demo.

---

## Rich search

`DOCUMENT.SEARCH` returns a flat `list[dict]` by default. Pass `"rich": True` (op dict) or `rich=True` (pipeline method) to get a structured response with provenance, neighbor context, and `by_document` grouping — the same KNN query with three extra layers on top.

**1. Provenance block** — provenance fields are packaged per result instead of scattered across the raw chunk dict:

```python
result["provenance"] == {
    "document_id": "doc_bc66f8ef9293d195",
    "file_name":   "annual_report.pdf",
    "chunk_id":    "doc_bc66f8ef9293d195_chunk_000003",
    "chunk_index": 3,
    "page":        4,                    # PDF only
    "section":     "2. Coverage",        # when SectionChunker detected a heading
}
```

**2. Neighbor context** — adjacent chunks from the same document, controlled by `context_chunks` (default `1`):

```python
result["context"] == {
    "before": "…text of the chunk just before this one…",
    "after":  "…text of the chunk just after this one…",
}
```

Set `context_chunks=0` to omit context entirely. Neighbor lookup is **in-process only** — the chunk index is built during `ingest()` and is lost when the process exits. Results for chunks not in the index silently omit the `"context"` key.

**3. `by_document` grouping** — results reorganised per source document with match count and deduplicated section titles:

```python
rich["by_document"] == {
    "doc_bc66f8ef9293d195": {
        "document_id":  "doc_bc66f8ef9293d195",
        "file_name":    "annual_report.pdf",
        "match_count":  3,
        "top_sections": ["2. Coverage", "4. Cost Sharing"],
        "matches":      […],   # same dicts as rich["results"]
    },
    …
}
```

**Usage:**

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(embedding_model=..., vector_source=...)
pipeline.ingest("policy.pdf")
pipeline.ingest("agreement.pdf")

# Pipeline call — opt in with rich=True
rich = pipeline.search("termination clause", k=5, context_chunks=1, rich=True)

print(rich["query"])           # original query string
print(rich["result_count"])    # number of matched chunks
print(rich["document_count"])  # number of distinct source documents

for r in rich["results"]:
    cit = r["provenance"]
    print(f"rank {r['rank']}  {cit['file_name']}  §{cit.get('section', '')}  p{cit.get('page', '')}")
    print(f"  {r['text'][:120]}")
    ctx = r.get("context", {})
    if ctx.get("before"):
        print(f"  ← {ctx['before'][:60]}")
    if ctx.get("after"):
        print(f"  → {ctx['after'][:60]}")

for doc_id, group in rich["by_document"].items():
    print(f"{group['file_name']}  hits={group['match_count']}")
    print(f"  sections: {group.get('top_sections', [])}")
```

Via `DocumentDataSource`:

```python
rich = source.execute_query({
    "op":            "DOCUMENT.SEARCH",
    "query":         "termination clause",
    "collection":    "contracts",
    "k":             5,
    "rich":          True,
    "context_chunks": 1,
})
```

| When to use `rich=True` | Reason |
|---|---|
| Building a provenance-aware UI | `page`, `section` are pre-extracted per result |
| Agent responses that cite sources | Provenance block is ready to embed in the prompt |
| Multi-document corpus | `by_document` shows which files matched and how many times |
| Feeding text to an LLM | Use plain search — rich overhead is wasted |

See `examples/feature-demos/rich_search_demo.py` for a runnable side-by-side comparison with captured output.

---

## Graph backend — entity-centric retrieval and structural navigation

Pass any blue-data `GraphSource` as `graph_source=` to populate a graph automatically during every ingest. The graph layer is optional — omit `graph_source` and the pipeline is unchanged.

> **What the graph is — and isn't.** This is an **entity / structure / provenance graph**, not a semantic knowledge graph. It captures document structure (`HAS_SECTION`, `HAS_CHUNK`, `NEXT`/`PREV` reading order), entity mentions (`MENTIONS`, `APPEARS_IN`), and entity co-occurrence (`CO_OCCURS_WITH`) — using the entities extracted by the enricher. It does **not** index the subject/predicate/object relationships that `LLMEnricher` extracts: those are stored on each chunk's `semantic_hints["relationships"]`, but are not written as graph edges. Indexing them as typed edges (with provenance and confidence) is future work (#65).

### Full graph schema

```
(Document {document_id, file_name, document_type, mime_type, ingested_at, chunk_count})
    -[:HAS_SECTION]->
(Section {section_id, title, document_id, position})
    -[:HAS_CHUNK]->
(Chunk {chunk_id, document_id, text_preview, chunk_index, page?, section_title?})
    -[:MENTIONS]->
(Entity {name, entity_type})

Additional edges:
  Document  -[:HAS_CHUNK]->       Chunk    (direct, regardless of section)
  Chunk     -[:NEXT]->            Chunk    (sequential forward reading order)
  Chunk     -[:PREV]->            Chunk    (sequential backward)
  Entity    -[:CO_OCCURS_WITH]->  Entity   (pairs co-occurring in the same chunk)
  Entity    -[:APPEARS_IN]->      Document (direct document-level presence — one-hop)
```

Node IDs are deterministic: `doc:{document_id}`, `section:{document_id}:{title.lower()}`, `chunk:{chunk_id}`, `entity:{entity_type}:{name.lower()}`. Re-ingesting the same file is a no-op for the graph (Document node exists → skip).

### Design notes

**`Document -[:HAS_CHUNK]->` exists alongside `Section -[:HAS_CHUNK]->`** so you can retrieve all chunks for a document in one hop without knowing whether sections exist. The section path is for structure-aware traversal; the direct path is for bulk access.

**`CO_OCCURS_WITH` edges** connect every pair of entities that appear in the same chunk. This lets you ask "what other entities tend to appear alongside Acme Corp?" as a graph query rather than a full-corpus text scan.

**`APPEARS_IN` edges** go directly from Entity to Document (not via Chunk). This means "which documents mention Acme Corp?" is a single hop, regardless of how many chunks in the document mention it. The edge is written once per document per entity, not once per chunk.

**`graph_neighbors()` uses a deterministic entity node ID** — constructed as `entity:{entity_type}:{name.lower()}` — so finding the entity node is a single dict lookup rather than a node scan. The subsequent edge traversal to collect all MENTIONS edges is proportional to the number of entity mentions, not total corpus size. No embeddings, no scoring.

**Entity nodes survive `delete_document()`** because they are shared across documents. Deleting one document must not break entity lookups that other documents depend on. Only Document, Section, and Chunk nodes are removed on delete.

### Setup

```python
from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import DocumentPipeline, BasicEnricher

graph = MemoryGraphSource("prod")
pipeline = DocumentPipeline(
    enricher=BasicEnricher(),   # extracts entities added as graph nodes
    graph_source=graph,
    graph_database="documents",
)
pipeline.ingest_text(
    "Acme Corp and Sarah Chen co-signed the agreement on January 15, 2026.",
    file_name="agreement.txt",
)
```

### Entity-centric retrieval — `graph_neighbors()`

Exact lookup by entity name — no embeddings, no scoring. Entity node ID is constructed deterministically, so the node is found in a single lookup. Results are proportional to the number of mentions, not total corpus size.

```python
# Find all chunks that mention "Acme Corp" across the entire corpus
results = pipeline.graph_neighbors("Acme Corp")
# → [{"chunk_id": "...", "document_id": "...", "text_preview": "...",
#     "entity": "Acme Corp", "entity_type": "PROPER_NOUN"}, ...]

# Restrict to one entity type
results = pipeline.graph_neighbors("January 15, 2026", entity_type="DATE")
```

Via `DocumentDataSource`:

```python
source.execute_query({
    "op": "DOCUMENT.GRAPH_NEIGHBORS",
    "entity": "Acme Corp",
    "entity_type": "PROPER_NOUN",  # optional
    "k": 10,
})
```

### General graph traversal — `graph_query()` / `DOCUMENT.GRAPH_QUERY`

Start from any node ID and follow any edge type in any direction.

```python
# All chunks directly owned by a document
pipeline.graph_query("doc:doc_abc123", edge_type="HAS_CHUNK", direction="out")

# Documents that mention an entity directly (APPEARS_IN — one hop)
pipeline.graph_query(
    "entity:PROPER_NOUN:acme corp",
    edge_type="APPEARS_IN",
    direction="out",
)

# Walk reading order — next chunk
pipeline.graph_query(
    "chunk:doc_abc123_chunk_000003",
    edge_type="NEXT",
    direction="out",
    k=1,
)

# Sections in a document
pipeline.graph_query("doc:doc_abc123", edge_type="HAS_SECTION", direction="out")
```

Via `DocumentDataSource`:

```python
source.execute_query({
    "op": "DOCUMENT.GRAPH_QUERY",
    "node_id": "entity:PROPER_NOUN:acme corp",
    "edge_type": "APPEARS_IN",
    "direction": "out",
})
```

Returns a list of dicts: `{"node_id", "label", "properties", "edge_type", "direction"}`.

### Backends

| Backend | Use case |
|---|---|
| `MemoryGraphSource` | Dev/test — in-process, no infra required |
| `FileGraphSource` | Local persistence |
| `NEO4JSource` | Production — Bolt protocol, Cypher, full graph query language |

Swap backends by changing the `graph_source=` argument — no other code changes.

```python
from blue_data.source.graph.neo4j.neo4j_graph_source import NEO4JSource

neo4j = NEO4JSource("neo4j", properties={
    "connection": {"host": "localhost", "port": 7687, "user": "neo4j", "password": "..."}
})
pipeline = DocumentPipeline(enricher=BasicEnricher(), graph_source=neo4j)
```

See `examples/feature-demos/graph_backend_demo.py` for a full runnable demo.

---

## Agent-friendly platform interfaces

The pipeline is accessible through four interfaces. All share the same environment variables and underlying `DocumentDataSource` — switching between them requires no code changes to the pipeline itself.

| Interface | Use case |
|---|---|
| Python library | Direct integration in any Python application or agent |
| MCP server | AI agents via Model Context Protocol (Claude Desktop, Claude Code, custom agents) |
| REST API | Non-Python services, microservice architectures, HTTP-based tool use |
| CLI | Interactive development, scripting, CI pipelines |

### MCP server

`platform/doc-mcp` exposes `DocumentDataSource` as tools for Claude and other MCP clients. Each tool maps to a `DOCUMENT.*` op and returns structured JSON.

```bash
cd platform/doc-mcp
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
    "blue-data-doc": {
      "command": "python",
      "args": ["/path/to/platform/doc-mcp/src/server.py"],
      "env": {
        "BLUE_DOC_EMBEDDING": "openai",
        "OPENAI_API_KEY": "sk-...",
        "BLUE_DOC_REDIS_HOST": "localhost"
      }
    }
  }
}
```

Tools exposed: `ensure_index`, `ingest_document`, `search_documents`, `delete_document`.

See [`docs/mcp.md`](mcp.md) for the full tool reference and environment variable documentation.

### REST API

`platform/doc-api` is a FastAPI service that exposes every `DOCUMENT.*` op as an HTTP endpoint. Useful for non-Python services, microservice architectures, or anywhere HTTP-based tool use is preferred.

```bash
cd platform/doc-api
BLUE_DOC_EMBEDDING=openai OPENAI_API_KEY=sk-... uvicorn doc_api.main:app --port 8001
```

| Endpoint | Method | Op equivalent |
|---|---|---|
| `/documents/health` | GET | — |
| `/documents/ensure-index` | POST | `DOCUMENT.ENSURE_INDEX` |
| `/documents/ingest` | POST | `DOCUMENT.INGEST` |
| `/documents/search` | POST | `DOCUMENT.SEARCH` |
| `/documents/find` | POST | `DOCUMENT.FIND` |
| `/documents/find-then-search` | POST | `DOCUMENT.FIND_THEN_SEARCH` |
| `/documents/record/{document_id}` | GET | `DOCUMENT.GET_RECORD` |
| `/documents/document` | DELETE | `DOCUMENT.DELETE` |

### CLI

`platform/doc-cli` provides a command-line interface for ingestion, search, and document management — useful for interactive development, shell scripts, and CI pipelines.

```bash
pip install -e platform/doc-cli

# Ingest a document
blue-data-doc ingest path/to/policy.pdf --collection contracts

# Search
blue-data-doc search "termination clause" --collection contracts -k 5

# Delete
blue-data-doc delete --path path/to/policy.pdf --collection contracts

# Output raw JSON for scripting
blue-data-doc search "deductible" --json
```

Configure via the same environment variables used by the MCP server and REST API (`BLUE_DOC_EMBEDDING`, `BLUE_DOC_REDIS_HOST`, `BLUE_DOC_COLLECTION`, etc.).
