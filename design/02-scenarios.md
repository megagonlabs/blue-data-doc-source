# blue-data-doc-source — Scenarios

## Scenario 1: Inspecting Chunks from a Document

A developer wants to understand how a policy PDF will be chunked before wiring up embeddings or a vector store. They run the pipeline with no embedding model and inspect the chunk records returned directly.

**Example:**

```python
from blue_data_doc_source import DocumentPipeline

pipeline = DocumentPipeline(chunk_size=500, overlap=100)
records = pipeline.ingest("data/sample_policy.pdf")

for r in records:
    print(r["chunk_id"], r.get("section_title"), len(r["text"].split()), "words")
```

Output:
```
doc_3a7f_chunk_000000  Eligibility Requirements   312 words
doc_3a7f_chunk_000001  Eligibility Requirements   287 words
doc_3a7f_chunk_000002  Coverage Terms             498 words
```

No API key, no vector store, no network calls. Each record carries full provenance: `document_id`, `chunk_id`, `file_name`, `page_start`, `page_end`, and `section_title`.

---

## Scenario 2: Full Pipeline — Ingest, Store, and Search

A developer builds a document retrieval system over a corpus of insurance contracts. They parse, chunk, enrich, embed, and upsert into Redis, then run a semantic search.

**Example:**

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, BasicEnricher
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=RedisVectorSource(name="contracts", host="localhost", port=6379),
    enricher=BasicEnricher(),
    chunk_size=500,
    overlap=100,
    default_collection="contracts",
)

source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})

for path in ["contract_a.pdf", "contract_b.pdf", "contract_c.pdf"]:
    records = source.execute_query({"op": "DOCUMENT.INGEST", "path": path, "collection": "contracts"})
    print(f"{path}: {len(records)} chunks")

results = source.execute_query({
    "op": "DOCUMENT.SEARCH",
    "query": "termination clause notice period",
    "collection": "contracts",
    "k": 5,
})
for r in results:
    print(r["file_name"], r.get("section_title"), r["text"][:100])
```

The same result is returned whether the data was ingested in this process or a previous one — the vector store is the source of truth.

---

## Scenario 3: Re-ingestion and Deduplication

A document corpus is updated periodically. A developer re-runs ingestion over the same directory — updated files are re-chunked and their old chunks replaced; unchanged files are upserted in place without creating duplicates.

**Example:**

```python
import os

for filename in os.listdir("contracts/"):
    records = source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": f"contracts/{filename}",
        "collection": "contracts",
    })
    print(f"{filename}: {len(records)} chunks (document_id: {records[0]['document_id']})")
```

Because `document_id` is a SHA-256 hash of the file bytes, running this twice on an unchanged file produces the same chunk IDs and the vector store upsert is a no-op at the data level. A modified file produces a new `document_id` — the pipeline automatically deletes the old chunks and writes the new ones (within the same process; cross-process deletion uses `DOCUMENT.DELETE`).

---

## Scenario 4: Scanned PDF with OCR

A developer ingests a batch of scanned insurance claim forms that have no embedded text layer. They enable OCR fallback so pages are processed by Tesseract instead of being silently skipped.

**Example:**

```python
from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

pipeline = DocumentPipeline(
    ocr_fallback=True,   # requires pip install blue-data-doc-source[ocr] + tesseract
    embedding_model=DummyEmbeddingModel(dim=8),
    vector_source=MemoryVectorSource(),
    collection="claims",
)
pipeline.ensure_index("claims")
records = pipeline.ingest("scanned_claim_form.pdf")

for r in records:
    ocr_flag = r["metadata"].get("ocr", False)
    confidence = r["metadata"].get("ocr_confidence")
    print(f"page {r['page_start']}: ocr={ocr_flag}, confidence={confidence:.1f}%")
```

OCR pages carry `metadata["ocr"] = True` and `metadata["ocr_confidence"]` (Tesseract mean word confidence, 0–100). Downstream code can use the confidence score to flag low-quality extractions for human review.

---

## Scenario 5: Custom Parser — LlamaIndex Integration

A developer needs to handle complex multi-column PDFs that the built-in `PDFParser` cannot layout correctly. They plug in LlamaIndex's PDF reader as a custom `Parser` — all downstream stages (chunking, enrichment, embedding, storage) are unchanged.

**Example:**

```python
from blue_data_doc_source import DocumentPipeline, Parser, OpenAIEmbeddingModel
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

class LlamaIndexParser(Parser):
    def parse(self, path: str):
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return [{"text": doc.text} for doc in docs]

pipeline = DocumentPipeline(
    parser=LlamaIndexParser(),
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    collection="reports",
)
pipeline.ensure_index("reports")
pipeline.ingest("complex_layout.pdf")
```

LlamaIndex handles parsing; the doc-source pipeline handles everything from chunking onward.

---

## Scenario 6: Pre-extracted Text from a Vision Model

A developer uses a vision model (e.g. GPT-4o with vision) to extract text from photographed documents. The text arrives as strings — not files — so they bypass the file parser entirely and feed it directly into the chunking → embedding → storage pipeline.

**Example:**

```python
# vision_model_output: list of {page_num: int, text: str} from an external tool
vision_model_output = [
    {"page_num": 1, "text": "INSURANCE CLAIM FORM\n\nClaimant: Alice Smith..."},
    {"page_num": 2, "text": "Section 3: Damages\n\nTotal claimed: $4,200..."},
]

records = pipeline.ingest_pages(
    vision_model_output,
    file_name="claim_form_alice.pdf",
    collection="claims",
)
```

`page_num` values propagate to `page_start` / `page_end` on each `ChunkRecord`, preserving page-level provenance even when the file was never parsed by the pipeline.

---

## Scenario 7: Natural Language Queries

A developer wants to let end users interact with the document store using plain English, without exposing the op interface. They wire up `NLQueryParser` backed by GPT-4o-mini to translate instructions into `DOCUMENT.*` ops automatically.

`NLQueryParser` maps intent to six op types: `DOCUMENT.SEARCH` (content retrieval), `DOCUMENT.FIND` (list/filter documents), `DOCUMENT.FIND_THEN_SEARCH` (two-part queries that identify a document subset and retrieve text from it), `DOCUMENT.INGEST`, `DOCUMENT.DELETE`, and `DOCUMENT.ENSURE_INDEX`.

**Example:**

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

source = DocumentDataSource(
    name="legal_corpus",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    nl_parser=NLQueryParser(),
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    default_collection="legal_corpus",
)

# DOCUMENT.SEARCH — content retrieval, no document filter
results = source.execute_query_nl("find clauses about termination notice periods")

# DOCUMENT.FIND — list documents, no content retrieval
records = source.execute_query_nl("list all contracts governed by California law")

# DOCUMENT.FIND_THEN_SEARCH — two-part: identify document subset, then retrieve text
chunks  = source.execute_query_nl("find the indemnification clause in Acme Corp contracts")

# DOCUMENT.INGEST
source.execute_query_nl("ingest /docs/new_contract.pdf into the legal_corpus collection")
```

The `NLQueryParser` accepts a custom `parse_fn` to swap in any LLM backend — Anthropic, a local model, or a test stub. The decision rules between FIND, FIND_THEN_SEARCH, and SEARCH are encoded in the system prompt — no application-level routing logic needed.

---

## Scenario 8: MCP Access from Claude Desktop

A team exposes the document store as an MCP server so Claude Desktop can ingest and search documents without any Python code. The `document-mcp` platform component wraps `DocumentDataSource` as a set of MCP tools.

**Example Claude Desktop config:**

```json
{
  "mcpServers": {
    "doc-source": {
      "command": "python",
      "args": ["-m", "platform.document-mcp.src.server"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379"
      }
    }
  }
}
```

Claude can then:
- Ingest a document: `"ingest /docs/policy.pdf into the contracts collection"`
- Search: `"find everything about payment obligations in contracts"`
- Delete: `"remove the old contract doc_abc123 from contracts"`

All pipeline logic — chunking, embedding, deduplication — runs server-side. Claude interacts only through MCP tool calls.

---

## Scenario 9: Structured Corpus Query (DOCUMENT.FIND + DOCUMENT.FIND_THEN_SEARCH)

A legal team has ingested 200 contracts, invoices, and policies into one corpus. They need to find all contracts governed by California law and retrieve their indemnification clauses — without the noise of semantic search returning irrelevant document types.

**Two-step pattern (explicit):**

```python
from blue_data_doc_source import (
    DocumentDataSource, OpenAIEmbeddingModel, KeywordClassifier,
    LLMFieldExtractor, FileRecordStore,
)
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

source = DocumentDataSource(
    name="legal_corpus",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=RedisVectorSource(host="localhost", port=6379),
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=FileRecordStore("schemas.json"),
    default_collection="legal_corpus",
)

# Step 1 — structured filter: no embedding, no LLM, deterministic
contracts = source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {
        "document_type": "contract",
        "fields.governing_law": {"contains": "California"},
    },
})
print(f"Found {len(contracts)} California contracts")
# → Found 3 California contracts

# Step 2 — text retrieval scoped to each matched document
for doc in contracts:
    clauses = source.execute_query({
        "op": "DOCUMENT.SEARCH",
        "query": "indemnification obligations",
        "filter": {"document_id": doc["document_id"]},
        "k": 2,
    })
    print(f"\n{doc['fields'].get('parties', [])}")
    for c in clauses:
        print(f"  § {c.get('section_title')}  {c['text'][:100]}")
```

**Same result in one call using `DOCUMENT.FIND_THEN_SEARCH`:**

```python
clauses = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {
        "document_type": "contract",
        "fields.governing_law": {"contains": "California"},
    },
    "query": "indemnification obligations",
    "k": 6,
})
# Returns merged, ranked chunks from all matched California contracts.
# Deduplication, merge, and sort are handled internally.
```

`DOCUMENT.FIND_THEN_SEARCH` accepts the same `find_filter` dict as `DOCUMENT.FIND`, the same `query` and `k` as `DOCUMENT.SEARCH`, and an optional `mode` (`"vector"`, `"bm25"`, `"hybrid"`, `"rerank"`). When `find_filter` matches no documents it returns `[]` immediately.

Without `DOCUMENT.FIND` / `DOCUMENT.FIND_THEN_SEARCH`, a search for "indemnification California" returns chunks from all document types, ranked by semantic similarity, with no reliable way to know which contract they belong to.

---

## Scenario 10: Persistent Schema Store Across Restarts

A developer runs nightly ingestion jobs that add new documents to a corpus. Agents querying the corpus the next morning need access to extracted schema fields without re-extracting them.

**Example:**

```python
# --- Ingest job (runs nightly) ---
from blue_data_doc_source import DocumentPipeline, FileRecordStore, KeywordClassifier, LLMFieldExtractor

pipeline = DocumentPipeline(
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=FileRecordStore("/data/schemas.json"),  # atomic write on every ingest
    embedding_model=...,
    vector_source=...,
)
pipeline.ingest("new_contract.pdf")
# → schemas.json updated atomically; no data loss on crash

# --- Agent query (next morning, new process) ---
pipeline2 = DocumentPipeline(
    classifier=KeywordClassifier(),
    record_store=FileRecordStore("/data/schemas.json"),  # reloads existing schemas on init
)

# All previously extracted schemas are immediately queryable
docs = pipeline2.find_records({
    "document_type": "contract",
    "fields.effective_date": {"gte": "2026-01-01"},
})
print(f"{len(docs)} contracts effective in 2026")
# → No re-extraction. No API calls. Just a filter over the loaded store.
```

`FileRecordStore` writes atomically (write-to-tmp, then `os.replace`) — a crash mid-write never corrupts the file. The store reloads cleanly from disk on every new process.

---

## Scenario 11: Compound Filter-then-Retrieve (DOCUMENT.FIND_THEN_SEARCH)

A developer is building a contract intelligence tool. Users want to ask things like "what does the Acme Corp agreement say about liability caps?" — which requires first identifying the right document, then retrieving the relevant passage. They want a single call rather than writing a find-then-loop pattern themselves.

**Example:**

```python
from blue_data_doc_source import (
    DocumentDataSource, OpenAIEmbeddingModel, KeywordClassifier,
    LLMFieldExtractor, FileRecordStore,
)
from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=RedisVectorSource(host="localhost", port=6379),
    classifier=KeywordClassifier(),
    field_extractor=LLMFieldExtractor(),
    record_store=FileRecordStore("schemas.json"),
    default_collection="contracts",
)

# Single call: find contracts mentioning Acme Corp, retrieve liability clauses from them
results = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {
        "document_type": "contract",
        "fields.parties": {"contains": "Acme Corp"},
    },
    "query": "liability cap limitation of liability",
    "k": 5,
    "mode": "hybrid",     # optional: "vector" | "bm25" | "hybrid" | "rerank"
})

for r in results:
    print(r["file_name"], r.get("section_title"), r["score"])
    print(r["text"][:150])
    print()
```

The op returns the same chunk dict format as `DOCUMENT.SEARCH`. When `find_filter` matches no documents, it returns `[]` immediately without touching the vector store. The merge step deduplicates and sorts by score across all matched documents.

For natural language dispatch, `NLQueryParser` emits `DOCUMENT.FIND_THEN_SEARCH` when the intent is both document-selection and content-retrieval in the same phrase:

```python
results = source.execute_query_nl(
    "find the liability cap clauses in Acme Corp contracts"
)
```

---

## Scenario 12: User-Defined Document Types via `extra_types`

A developer is building a tool for a property management company. Their corpus mixes standard contracts with lease agreements and employment contracts — document types the built-in `KeywordClassifier` does not recognise. They want to classify and filter these types the same way built-in types work, without forking the library.

**Example:**

```python
from blue_data_doc_source import (
    DocumentDataSource, KeywordClassifier, LLMFieldExtractor, FileRecordStore,
    DummyEmbeddingModel,
)
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

# Register custom types alongside built-ins — no subclassing required
classifier = KeywordClassifier(
    extra_types={
        "lease": [
            "lease agreement", "landlord", "tenant", "monthly rent",
            "security deposit", "premises", "lease term", "lessor", "lessee",
        ],
        "employment_contract": [
            "employer", "employee", "salary", "start date", "job title",
            "at-will employment", "non-compete", "severance",
        ],
    }
)

# Custom field specs for LLMFieldExtractor
extractor = LLMFieldExtractor(
    extra_types={
        "lease": (
            "Extract: monthly_rent (number), lease_term_months (int), "
            "landlord (string), tenant (string), property_address (string), "
            "security_deposit (number)"
        ),
        "employment_contract": (
            "Extract: employer (string), employee (string), job_title (string), "
            "salary (number), start_date (string), non_compete (bool)"
        ),
    }
)

source = DocumentDataSource(
    name="property_corpus",
    embedding_model=DummyEmbeddingModel(dim=8),
    vector_source=MemoryVectorSource(),
    classifier=classifier,
    field_extractor=extractor,
    record_store=FileRecordStore("schemas.json"),
    default_collection="property_corpus",
)

source.execute_query({"op": "DOCUMENT.ENSURE_INDEX"})
source.execute_query({"op": "DOCUMENT.INGEST", "path": "lease_apartment_4b.pdf"})
source.execute_query({"op": "DOCUMENT.INGEST", "path": "employment_alice.pdf"})

# Filter by custom type — identical API to built-in types
leases = source.execute_query({
    "op": "DOCUMENT.FIND",
    "filter": {
        "document_type": "lease",
        "fields.monthly_rent": {"gt": 2000},
    },
})
print(f"Leases > $2000/month: {len(leases)}")

# FIND_THEN_SEARCH works with custom types too
clauses = source.execute_query({
    "op": "DOCUMENT.FIND_THEN_SEARCH",
    "find_filter": {"document_type": "lease"},
    "query": "early termination penalty",
    "k": 4,
})
```

`extra_types` entries are merged at init time — built-in types (`contract`, `invoice`, `policy`, `research_paper`, `financial_report`) coexist with user-defined types. If a user-defined name collides with a built-in, the user-defined entry takes precedence. No restart or subclassing required.

---

## Scenario 13: Entity-Centric Retrieval via Graph Backend

A compliance team has 300 documents — contracts, vendor agreements, and incident reports — all mentioning overlapping entities: company names, people, dates. They need to answer questions like "which documents mention Acme Corp?" and "show me every chunk where Sarah Chen appears, with the surrounding chunks for context" — without relying on embedding similarity.

Vector search embeds the entity name and returns the k most semantically similar chunks. This misses exact mentions in short passages or tables where the entity name appears without rich surrounding prose, and cannot answer "how many documents mention this entity" without reading all results.

**Solution — graph backend:**

```python
from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import DocumentPipeline, BasicEnricher

graph = MemoryGraphSource("compliance")
pipeline = DocumentPipeline(
    enricher=BasicEnricher(),   # extracts PROPER_NOUN and DATE entities
    graph_source=graph,
    graph_database="documents",
)

# Ingest — graph nodes and edges created automatically alongside chunks
for path in document_paths:
    pipeline.ingest(path)

# exact entity lookup — no corpus scan, no embeddings; returns all mentions across all 300 docs
results = pipeline.graph_neighbors("Acme Corp")
print(f"Found in {len({r['document_id'] for r in results})} documents")

# Walk reading order — expand context around a hit via NEXT/PREV edges
chunk_id = results[0]["chunk_id"]
node_id  = f"chunk:{chunk_id}"

next_chunk = pipeline.graph_query(node_id, edge_type="NEXT", direction="out", k=1)
prev_chunk = pipeline.graph_query(node_id, edge_type="PREV", direction="out", k=1)

# Which documents does "Acme Corp" appear in directly — one hop via APPEARS_IN
doc_refs = pipeline.graph_query(
    "entity:PROPER_NOUN:acme corp",
    edge_type="APPEARS_IN",
    direction="out",
)
print(f"Acme Corp directly appears in {len(doc_refs)} document(s):")
for ref in doc_refs:
    print(f"  {ref['node_id']}  ({ref['properties'].get('file_name')})")
```

**Via `DocumentDataSource` ops:**

```python
# Entity-centric chunk retrieval
results = source.execute_query({
    "op": "DOCUMENT.GRAPH_NEIGHBORS",
    "entity": "Acme Corp",
    "entity_type": "PROPER_NOUN",
    "k": 50,
})

# General traversal — sections in a document
source.execute_query({
    "op": "DOCUMENT.GRAPH_QUERY",
    "node_id": "doc:doc_abc123",
    "edge_type": "HAS_SECTION",
    "direction": "out",
})

# Next chunk in reading order
source.execute_query({
    "op": "DOCUMENT.GRAPH_QUERY",
    "node_id": "chunk:doc_abc123_chunk_000003",
    "edge_type": "NEXT",
    "direction": "out",
    "k": 1,
})
```

**Production — swap to Neo4j with no code change:**

```python
from blue_data.source.graph.neo4j.neo4j_graph_source import NEO4JSource

neo4j = NEO4JSource("neo4j", properties={
    "connection": {"host": "localhost", "port": 7687, "user": "neo4j", "password": "..."}
})
pipeline = DocumentPipeline(enricher=BasicEnricher(), graph_source=neo4j)
```

The same `GraphIndexer`, same node IDs (`doc:`, `section:`, `chunk:`, `entity:`), same edges. No other changes.

Key properties:
- **Idempotent** — re-ingesting the same file is a no-op (Document node already exists)
- **Shared entity nodes** — "Acme Corp" is one node connected to every chunk and document that mentions it; updating the entity's properties propagates everywhere
- **Delete cascade** — `DOCUMENT.DELETE` removes the Document, its Section nodes, and its Chunk nodes; Entity nodes are retained because they may be referenced by other documents
- **Zero overhead when unused** — `graph_source=None` (the default) adds no overhead to any pipeline operation
