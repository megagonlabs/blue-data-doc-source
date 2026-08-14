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

**Example:**

```python
from blue_data_doc_source import DocumentDataSource, OpenAIEmbeddingModel, NLQueryParser
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

source = DocumentDataSource(
    name="contracts",
    embedding_model=OpenAIEmbeddingModel(),
    vector_source=MemoryVectorSource(),
    nl_parser=NLQueryParser(),
    default_collection="contracts",
)

# NL instructions — the LLM decides the op and parameters
results = source.execute_query_nl("find clauses about termination notice periods")
records = source.execute_query_nl("ingest /docs/new_contract.pdf into the contracts collection")
```

The `NLQueryParser` accepts a custom `parse_fn` to swap in any LLM backend — Anthropic, a local model, or a test stub.

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
