"""
DOCUMENT.FIND — Structured Document Filtering
=============================================
Shows how DOCUMENT.FIND queries across ingested documents by document_type
and extracted field values — with no embedding, no LLM at query time.

The problem it solves:
  In a mixed corpus, semantic search cannot answer structural questions:
    "Give me all contracts" — embedding can't enumerate document types
    "Which invoices exceed $30k?" — embeddings can't compare numbers
    "Find contracts governed by California" — "California" appears in many docs

DOCUMENT.FIND evaluates deterministic filters over DocumentRecords written
at ingest time. Each filter condition is ANDed. Evaluation is O(n) over
stored records — no vector index, no query-time API call.

Filter syntax:
  {"document_type": "contract"}
  {"fields.governing_law": {"contains": "California"}}
  {"fields.total_due": {"gt": 30000}}      # "$35,154" → 35154 > 30000
  {"document_type": "invoice", "fields.total_due": {"gte": 5000}}

Run (no API key needed — type filtering works with KeywordClassifier):
    python3 examples/feature-demos/document_find_demo.py

Run (with LLM field extraction — enables field-level filters):
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/document_find_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import (
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    LLMFieldExtractor,
)

DIV  = "=" * 72
DIV2 = "-" * 72


def _banner(title: str) -> None:
    print(); print(DIV); print(title); print(DIV)


def _step(label: str) -> None:
    print(); print(DIV2); print(label); print(DIV2)


# ---------------------------------------------------------------------------
# Inline corpus — no external files needed
# ---------------------------------------------------------------------------

CORPUS = {
    "acme_contract.txt": """\
SERVICE AGREEMENT — ACME CORP

This Service Agreement is entered into between TechVendor Inc. and Acme Corp,
effective January 15, 2026. The governing law of this agreement is the State of
Delaware. The contract value is one hundred eighty thousand dollars per annum.

OBLIGATIONS
TechVendor Inc. shall deliver software consulting services as specified in the
Statement of Work. Acme Corp shall provide timely access to systems and personnel.
Both parties shall maintain confidentiality of proprietary information.

TERMINATION
Either party may terminate this agreement with sixty days written notice.
Acme Corp may terminate immediately upon material breach by TechVendor Inc.

GOVERNING LAW AND DISPUTES
This agreement is governed by the laws of the State of Delaware.
Any disputes shall be resolved through binding arbitration in Wilmington.
""",
    "beta_contract.txt": """\
CONSULTING AGREEMENT — BETA LTD

This Consulting Agreement is entered into between DataSystems LLC and Beta Ltd,
effective March 1, 2026. The governing law is the State of California. The total
contract value is forty-five thousand dollars for the project term.

TERMINATION
Either party may terminate this agreement with thirty days written notice.
Immediate termination is permitted in the event of insolvency or material breach.

GOVERNING LAW
This agreement is governed exclusively by the laws of the State of California.
Disputes shall be resolved in San Francisco County courts.
""",
    "invoice_aug26.txt": """\
INVOICE

Invoice Number: INV-2026-0891
Invoice Date: August 1, 2026
Due Date: August 31, 2026

Vendor: Blue Solutions Inc.
Bill To: Acme Manufacturing Corp

Line Items:
  Cloud Infrastructure Design & Architecture     $12,000
  Security Audit (40 hours at $250/hr)           $10,000
  Staff Training Workshop (3 groups)             $2,400
  Data Migration Planning and Documentation      $4,500
  Project Management (20 hours at $175/hr)       $3,500
  Performance Optimization Review                $2,754

Subtotal: $32,400
Tax (10%): $2,754
Total Due: $35,154

Payment Terms: Net 30 days. Late payments accrue interest at 1.5% per month.
""",
    "health_policy.txt": """\
HEALTH INSURANCE POLICY

Policy Number: BSH-2024-004521
Effective Date: January 1, 2024
Insured: Jane Doe

COVERAGE BENEFITS
This policy covers preventive care, primary care visits, specialist consultations,
emergency room visits, and prescription drugs.

DEDUCTIBLES AND OUT-OF-POCKET
The annual individual deductible is one thousand five hundred dollars.
After the deductible is met, the plan covers 80% of eligible costs.

EXCLUSIONS
This policy does not cover cosmetic procedures, experimental treatments,
dental and vision care, or injuries from professional sports.

POLICY TERMINATION
This policy may be terminated by the insurer with 30 days written notice
for non-payment of premiums. COBRA continuation is available for 18 months.
""",
    "hybrid_rank.txt": """\
HybridRank: Combining Sparse and Dense Retrieval

Authors: Alexandra Chen, Marcus Williams, Priya Nair

Abstract
We present HybridRank, a retrieval architecture that combines sparse BM25 ranking
with dense neural embeddings. Our findings demonstrate a 40% improvement over
baseline approaches on the MS MARCO and TriviaQA benchmarks.

Methodology
We use BM25 with parameters k1=1.6 and b=0.75 for the sparse stage.
Dense embeddings are produced by a fine-tuned bi-encoder. Reciprocal rank fusion
combines both ranked lists.

Keywords: hybrid retrieval, BM25, dense embeddings, information retrieval
""",
}


def run() -> None:
    _banner("DOCUMENT.FIND — Structured Document Filtering")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Corpus: 2 contracts, 1 invoice, 1 insurance policy, 1 research paper")
    if has_openai:
        print("  OPENAI_API_KEY detected — field-level filters enabled")
    else:
        print("  No OPENAI_API_KEY — document_type filters fully demonstrated")
        print("  Set OPENAI_API_KEY to enable field-level filters (governing_law, total_due, ...)")

    # ── Build pipeline ────────────────────────────────────────────────────
    field_extractor = LLMFieldExtractor() if has_openai else None
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=120,
        overlap=20,
        collection="find_demo",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    pipeline.ensure_index("find_demo")

    # ── Ingest corpus ─────────────────────────────────────────────────────
    _step("Ingest — one DocumentRecord written per document at ingest time")
    print()
    doc_ids: dict[str, str] = {}
    for fname, content in CORPUS.items():
        records = pipeline.ingest_text(
            content, file_name=fname, mime_type="text/plain", collection="find_demo"
        )
        doc_id     = records[0]["document_id"]
        doc_ids[fname] = doc_id
        record     = pipeline.get_record(doc_id)
        doc_type   = record["document_type"] if record else "unknown"
        confidence = f"{record['confidence']:.2f}" if record else "?"
        n_chunks   = len(records)
        print(f"  {fname:<26}  type={doc_type:<16}  confidence={confidence}  chunks={n_chunks}")

    print()
    print("  KeywordClassifier runs once per document at ingest — no API cost.")
    print("  Each DocumentRecord is stored in-process (MemoryRecordStore by default).")
    print("  LLMFieldExtractor runs after classification when OPENAI_API_KEY is set.")

    # ── Filter by document_type ───────────────────────────────────────────
    _step("Filter by document_type")
    print()
    print("  pipeline.find_records({'document_type': 'contract'})")
    print()
    print("  Returns every DocumentRecord where document_type == 'contract'.")
    print("  Deterministic. No embedding. No LLM at query time.")
    print()

    for doc_type in ["contract", "invoice", "policy", "research_paper"]:
        matches = pipeline.find_records({"document_type": doc_type})
        names   = [fname for fname, did in doc_ids.items()
                   if any(m["document_id"] == did for m in matches)]
        tick = "✓" if matches else "·"
        print(f"  {tick}  find_records({{'document_type': {doc_type!r}}})  ->  "
              f"{len(matches)} match(es)  {names}")

    # ── confidence filter ─────────────────────────────────────────────────
    _step("Filter by confidence score")
    print()
    print("  confidence is written by KeywordClassifier — no LLM needed.")
    print("  Use it to narrow to high-confidence classifications only.")
    print()

    for threshold in [0.3, 0.5, 0.7]:
        matches = pipeline.find_records({"confidence": {"gte": threshold}})
        print(f"  find_records({{'confidence': {{'gte': {threshold}}}}})  "
              f"->  {len(matches)} document(s)")
        for m in matches:
            print(f"      {m['document_type']:<18}  confidence={m['confidence']:.4f}")

    # ── Field-level filters (OpenAI) ─────────────────────────────────────
    _step("Filter by extracted field values")
    print()

    if has_openai:
        print("  LLMFieldExtractor populates fields at ingest time.")
        print("  DOCUMENT.FIND evaluates filters over these stored field values.")
        print()

        # governing_law
        print("  find_records({'fields.governing_law': {'contains': 'California'}}):")
        results = pipeline.find_records({"fields.governing_law": {"contains": "California"}})
        for r in results:
            print(f"    -> {r['document_type']:<18}  governing_law={r['fields'].get('governing_law')!r}")
        if not results:
            print("    (no matches — check that governing_law was extracted)")
        print()

        # total_due > 30000
        print("  find_records({'document_type': 'invoice', 'fields.total_due': {'gt': 30000}}):")
        print("    # '$35,154' is coerced to 35154 before comparison")
        results = pipeline.find_records({
            "document_type": "invoice",
            "fields.total_due": {"gt": 30000},
        })
        for r in results:
            print(f"    -> {r['document_type']:<18}  total_due={r['fields'].get('total_due')!r}"
                  f"  vendor={r['fields'].get('vendor')!r}")
        if not results:
            print("    (no matches — check that total_due was extracted)")
        print()

        # multiple conditions
        print("  find_records({'document_type': 'contract', 'confidence': {'gte': 0.5}}):")
        results = pipeline.find_records({"document_type": "contract", "confidence": {"gte": 0.5}})
        for r in results:
            print(f"    -> {r['document_type']:<18}  confidence={r['confidence']:.4f}"
                  f"  parties={r['fields'].get('parties')!r}")
    else:
        print("  Without OPENAI_API_KEY, fields dict is empty and field-level filters")
        print("  return no matches. document_type and confidence filters work without it.")
        print()
        print("  Filter operators available once fields are populated:")
        print()
        operators = [
            ("contains", "substring match (case-insensitive)",
             "{'fields.governing_law': {'contains': 'California'}}"),
            ("gt",  "greater than (numeric or lexicographic)",
             "{'fields.total_due': {'gt': 30000}}"),
            ("gte", "greater than or equal",
             "{'fields.total_due': {'gte': 30000}}"),
            ("lt",  "less than",
             "{'fields.total_due': {'lt': 100000}}"),
            ("lte", "less than or equal",
             "{'fields.total_due': {'lte': 100000}}"),
        ]
        print(f"  {'Operator':<10}  {'Meaning':<40}  Example filter")
        print(f"  {'-'*10}  {'-'*40}  {'-'*40}")
        for op, meaning, example in operators:
            print(f"  {op:<10}  {meaning:<40}  {example}")
        print()
        print("  Numeric coercion: strips $€£¥ and commas before comparison.")
        print("  All conditions in a filter dict are ANDed.")

    # ── DOCUMENT.FIND via DocumentDataSource ──────────────────────────────
    _step("DOCUMENT.FIND via DocumentDataSource op interface")
    print()
    print("  When using DocumentDataSource (MCP / registry / API), FIND is an op:")
    print()
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.FIND',")
    print("        'filter': {'document_type': 'contract'},")
    print("    })")
    print()

    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=120,
        overlap=20,
        default_collection="find_op_demo",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "find_op_demo"})

    import tempfile
    with tempfile.TemporaryDirectory() as _tmpdir:
        import os as _os
        for _fname, _content in CORPUS.items():
            _fpath = _os.path.join(_tmpdir, _fname)
            with open(_fpath, "w") as _f:
                _f.write(_content)
            source.execute_query({"op": "DOCUMENT.INGEST", "path": _fpath,
                                  "collection": "find_op_demo"})
        op_results = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "contract"},
        })

    print(f"  Result: {len(op_results)} DocumentRecord(s)")
    for r in op_results:
        print(f"    document_type={r['document_type']!r}  "
              f"document_id={r['document_id'][:20]}...  "
              f"confidence={r['confidence']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  DOCUMENT.FIND characteristics:")
    print()
    print("    Deterministic   — same filter always returns the same documents")
    print("    No embedding    — runs without an embedding model")
    print("    No query LLM    — field values are extracted once at ingest")
    print("    O(n) scan       — not indexed; suitable for corpora up to ~10k documents")
    print()
    print("  What it returns:")
    print("    One DocumentRecord per matched document — not chunks.")
    print("    Each record: document_id, document_type, confidence, classifier, fields")
    print()
    print("  Filter syntax (all conditions ANDed):")
    print("    {'document_type': 'contract'}                     exact match")
    print("    {'fields.governing_law': {'contains': 'CA'}}      substring")
    print("    {'fields.total_due': {'gt': 30000}}               numeric (strips $,£,€,¥,comma)")
    print("    {'confidence': {'gte': 0.5}}                      classifier confidence")
    print()
    print("  Next step after FIND — retrieve text from matched documents:")
    print("    docs   = pipeline.find_records({'document_type': 'contract'})")
    print("    chunks = pipeline.search('termination', filters={'document_id': docs[0]['document_id']})")
    print()
    print("  Or in one call — see find_then_search_demo.py:")
    print("    source.execute_query({'op': 'DOCUMENT.FIND_THEN_SEARCH', ...})")
    print()
    if not has_openai:
        print("  Enable field-level filters:")
        print("    OPENAI_API_KEY=sk-... python3 examples/feature-demos/document_find_demo.py")
        print()


if __name__ == "__main__":
    run()
