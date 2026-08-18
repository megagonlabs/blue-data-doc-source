"""
DOCUMENT.FIND_THEN_SEARCH — Compound Filter + Retrieval
=======================================================
Shows how DOCUMENT.FIND_THEN_SEARCH replaces a manual filter-then-search
loop with a single op call.

The problem:
  A mixed corpus of contracts, invoices, and policies. The agent needs the
  termination clause from contracts only. Two operations are required:
    1. DOCUMENT.FIND — filter DocumentRecords to get matching doc IDs
    2. DOCUMENT.SEARCH — retrieve text chunks scoped to those documents

Doing this manually means writing a loop, deduplicating results, and sorting
by score. DOCUMENT.FIND_THEN_SEARCH encapsulates all three steps.

What it does:
  1. Runs find_records(find_filter) — same as DOCUMENT.FIND
  2. For each matched document, runs search(query, filters={document_id: ...})
  3. Merges all chunk results, deduplicates, sorts by score, returns top k

Run (no API key needed):
    python3 examples/feature-demos/find_then_search_demo.py

Run (with LLM field extraction — enables field-level find_filter):
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/find_then_search_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import (
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    LLMFieldExtractor,
    NLQueryParser,
)

DIV  = "=" * 72
DIV2 = "-" * 72


def _banner(title: str) -> None:
    print(); print(DIV); print(title); print(DIV)


def _step(label: str) -> None:
    print(); print(DIV2); print(label); print(DIV2)


def _chunk_table(results: list, max_rows: int = 6) -> None:
    print(f"  {'score':<8}  {'file':<26}  {'section':<26}  text preview")
    print(f"  {'-'*8}  {'-'*26}  {'-'*26}  {'-'*30}")
    for r in results[:max_rows]:
        score   = f"{r.get('score') or 0:.4f}"
        fname   = r.get("file_name", "?")
        section = (r.get("section_title") or "(no section)")[:26]
        snippet = r.get("text", "").replace("\n", " ")[:35]
        print(f"  {score:<8}  {fname:<26}  {section:<26}  {snippet}...")


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
Upon termination, TechVendor Inc. shall deliver all work product completed to date.

GOVERNING LAW AND DISPUTES
This agreement is governed by the laws of the State of Delaware.
""",
    "beta_contract.txt": """\
CONSULTING AGREEMENT — BETA LTD

This Consulting Agreement is entered into between DataSystems LLC and Beta Ltd,
effective March 1, 2026. The governing law is the State of California. The total
contract value is forty-five thousand dollars for the project term.

TERMINATION
Either party may terminate this agreement with thirty days written notice.
Immediate termination is permitted in the event of insolvency or material breach.
DataSystems LLC shall retain fees for all work completed prior to termination.

GOVERNING LAW
This agreement is governed exclusively by the laws of the State of California.
""",
    "invoice_aug26.txt": """\
INVOICE

Invoice Number: INV-2026-0891
Vendor: Blue Solutions Inc.
Bill To: Acme Manufacturing Corp

Line Items:
  Cloud Infrastructure Design     $12,000
  Security Audit                  $10,000
  Staff Training Workshop          $2,400

Subtotal: $24,400
Tax: $2,440
Total Due: $26,840

Payment Terms: Net 30 days.
""",
    "health_policy.txt": """\
HEALTH INSURANCE POLICY

Policy Number: BSH-2024-004521
Insured: Jane Doe

COVERAGE BENEFITS
This policy covers preventive care, primary care visits, specialist consultations,
emergency room visits, and prescription drugs.

POLICY TERMINATION
This policy may be terminated by the insurer with 30 days written notice.
COBRA continuation coverage is available for 18 months following termination.
""",
}


def run() -> None:
    _banner("DOCUMENT.FIND_THEN_SEARCH — Compound Filter + Retrieval")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Corpus: 2 contracts, 1 invoice, 1 insurance policy")
    print("  Query : 'termination notice period' — contracts only")
    print()
    if has_openai:
        print("  OPENAI_API_KEY detected — field-level find_filter enabled")
    else:
        print("  No OPENAI_API_KEY — document_type filter used throughout")

    # ── Build pipeline ────────────────────────────────────────────────────
    field_extractor = LLMFieldExtractor() if has_openai else None
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=100,
        overlap=15,
        collection="fts_demo",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    pipeline.ensure_index("fts_demo")

    _step("Ingest corpus")
    print()
    for fname, content in CORPUS.items():
        records = pipeline.ingest_text(
            content, file_name=fname, mime_type="text/plain", collection="fts_demo"
        )
        record   = pipeline.get_record(records[0]["document_id"])
        doc_type = record["document_type"] if record else "unknown"
        print(f"  {fname:<26}  {len(records)} chunks  type={doc_type}")

    # ====================================================================
    # The manual two-step (what callers used to write)
    # ====================================================================
    _step("The manual two-step — verbose but equivalent")
    print()
    print("  Without DOCUMENT.FIND_THEN_SEARCH the caller writes a loop:")
    print()
    print("    matched = pipeline.find_records({'document_type': 'contract'})")
    print("    seen, all_results = set(), []")
    print("    for doc in matched:")
    print("        chunks = pipeline.search(query, filters={'document_id': doc['document_id']})")
    print("        for chunk in chunks:")
    print("            if chunk['chunk_id'] not in seen:")
    print("                seen.add(chunk['chunk_id'])")
    print("                all_results.append(chunk)")
    print("    all_results.sort(key=lambda c: c.get('score') or 0, reverse=True)")
    print("    return all_results[:k]")
    print()

    matched = pipeline.find_records({"document_type": "contract"})
    seen: set = set()
    manual_results: list = []
    for doc in matched:
        doc_id = doc["document_id"]
        chunks = pipeline.search(
            "termination notice period", k=5,
            collection="fts_demo",
            filters={"document_id": doc_id},
            mode="bm25",
        )
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                manual_results.append(chunk)
    manual_results.sort(key=lambda c: c.get("score") or 0, reverse=True)
    manual_results = manual_results[:5]

    print(f"  Manual two-step result: {len(manual_results)} chunk(s) from {len(matched)} contract(s)")
    print()
    _chunk_table(manual_results)

    # ====================================================================
    # DOCUMENT.FIND_THEN_SEARCH — one call
    # ====================================================================
    _step("DOCUMENT.FIND_THEN_SEARCH — same result, one call")
    print()
    print("  source.execute_query({")
    print("      'op': 'DOCUMENT.FIND_THEN_SEARCH',")
    print("      'find_filter': {'document_type': 'contract'},")
    print("      'query': 'termination notice period',")
    print("      'k': 5,")
    print("      'mode': 'bm25',")
    print("  })")
    print()
    print("  The op handles find → search-per-doc → merge → sort internally.")
    print()

    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=100,
        overlap=15,
        default_collection="fts_op_demo",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "fts_op_demo"})

    with tempfile.TemporaryDirectory() as _tmpdir:
        import os as _os
        for _fname, _content in CORPUS.items():
            _fpath = _os.path.join(_tmpdir, _fname)
            with open(_fpath, "w") as _f:
                _f.write(_content)
            source.execute_query({"op": "DOCUMENT.INGEST", "path": _fpath,
                                  "collection": "fts_op_demo"})
        fts_results = source.execute_query({
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": "contract"},
            "query": "termination notice period",
            "k": 5,
            "mode": "bm25",
            "collection": "fts_op_demo",
        })

    print(f"  Result: {len(fts_results)} chunk(s)")
    print()
    _chunk_table(fts_results)

    doc_ids_seen = {r.get("document_id") for r in fts_results}
    print()
    print(f"  Chunks from {len(doc_ids_seen)} matched document(s),"
          " merged and sorted by score.")
    print("  No duplicate chunks — deduplication is built in.")

    # ====================================================================
    # execute_query_nl dispatch
    # ====================================================================
    _step("execute_query_nl — NLQueryParser dispatches to FIND_THEN_SEARCH")
    print()
    print("  When NLQueryParser detects 'find documents of type X, then search")
    print("  for Y within them', it emits DOCUMENT.FIND_THEN_SEARCH automatically.")
    print()
    print("  source.execute_query_nl(")
    print("      'find the termination clause in our contracts'")
    print("  )")
    print()
    print("  Simulated parse result (real NLQueryParser uses GPT):")
    print("    op          = DOCUMENT.FIND_THEN_SEARCH")
    print("    find_filter = {'document_type': 'contract'}")
    print("    query       = 'termination notice period'")
    print()

    def _simulated_parse(text: str) -> dict:
        return {
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": "contract"},
            "query": "termination notice period",
            "k": 5,
            "mode": "bm25",
            "collection": "nl_fts_demo",
        }

    nl_source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=100,
        overlap=15,
        default_collection="nl_fts_demo",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
        nl_parser=NLQueryParser(parse_fn=_simulated_parse),
    )
    nl_source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "nl_fts_demo"})

    with tempfile.TemporaryDirectory() as _tmpdir:
        import os as _os
        for _fname, _content in CORPUS.items():
            _fpath = _os.path.join(_tmpdir, _fname)
            with open(_fpath, "w") as _f:
                _f.write(_content)
            nl_source.execute_query({"op": "DOCUMENT.INGEST", "path": _fpath,
                                     "collection": "nl_fts_demo"})
        nl_results = nl_source.execute_query_nl(
            "find the termination clause in our contracts"
        )

    print(f"  execute_query_nl(...)  ->  {len(nl_results)} chunk(s)")
    if nl_results:
        _chunk_table(nl_results)

    # ====================================================================
    # Field-level find_filter (OpenAI only)
    # ====================================================================
    if has_openai:
        _step("Field-level find_filter — compound filter + text retrieval")
        print()
        print("  find_filter accepts any filter DOCUMENT.FIND supports —")
        print("  including extracted field values:")
        print()
        print("    source.execute_query({")
        print("        'op': 'DOCUMENT.FIND_THEN_SEARCH',")
        print("        'find_filter': {")
        print("            'document_type': 'contract',")
        print("            'fields.governing_law': {'contains': 'California'},")
        print("        },")
        print("        'query': 'termination notice period',")
        print("        'k': 3,")
        print("    })")
        print()
        print("  This finds the one contract governed by California law, then")
        print("  retrieves termination clauses from that document only.")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  DOCUMENT.FIND_THEN_SEARCH replaces:")
    print()
    print("    matched = pipeline.find_records(find_filter)")
    print("    results = []")
    print("    for doc in matched:")
    print("        results += pipeline.search(query, filters={'document_id': doc['document_id']})")
    print("    results = sorted(set(results), key=score)[:k]")
    print()
    print("  With:")
    print()
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.FIND_THEN_SEARCH',")
    print("        'find_filter': {'document_type': 'contract'},")
    print("        'query': 'termination notice period',")
    print("        'k': 5,")
    print("    })")
    print()
    print("  Key properties:")
    print("    find_filter  — same filter dict as DOCUMENT.FIND (type, field values)")
    print("    query        — text searched within matched documents only")
    print("    k            — cap applied after merge across all matched documents")
    print("    mode         — 'vector' (default), 'bm25', 'hybrid', or 'rerank'")
    print("    collection   — vector store collection to search in")
    print()
    print("  Zero matches on find_filter returns [] immediately — no search runs.")
    print()
    if not has_openai:
        print("  Enable field-level find_filter:")
        print("    OPENAI_API_KEY=sk-... python3 examples/feature-demos/find_then_search_demo.py")
        print()


if __name__ == "__main__":
    run()
