"""
Document Schema Inference
=========================
Shows how KeywordClassifier identifies document types from PDF content and
LLMFieldExtractor extracts typed fields -- no manual labelling, no file
re-reading after ingest.

Story:
  A company receives a weekly drop of PDFs: contracts, invoices, research
  papers, and policy documents. The team wants to know what type each file is
  and pull out key fields (parties, due dates, policy numbers, ...) to power
  a rich search UI -- without paying per-chunk API costs.

Two-layer approach:
  Layer 1 -- KeywordClassifier (always runs, no API cost)
    Scores the first 500 words against signal phrase sets for each document
    type and returns (document_type, confidence). Zero dependencies.

  Layer 2 -- LLMFieldExtractor (optional, requires OPENAI_API_KEY)
    Calls GPT-4o-mini once per document with a type-specific prompt and
    extracts typed fields (parties, effective_date, invoice_number, ...).

What this demo covers:
  * pipeline.ingest(path)  -- real PDF files, page-by-page parsing
  * document_type on every chunk after ingest
  * pipeline.get_record(document_id)  -- schema stored at ingest time
  * confidence scores from KeywordClassifier
  * Typed fields from LLMFieldExtractor (when OPENAI_API_KEY is set)
  * DOCUMENT.GET_RECORD op via DocumentDataSource

Run:
    python3 examples/feature-demos/field_extraction_demo.py

    # With LLM field extraction:
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/field_extraction_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import (
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    LLMFieldExtractor,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DIV  = "=" * 72
DIV2 = "-" * 72

PDFS = [
    DATA_DIR / "sample_contract.pdf",
    DATA_DIR / "sample_invoice.pdf",
    DATA_DIR / "sample_research.pdf",
    DATA_DIR / "sample_policy.pdf",
]


def _banner(title: str) -> None:
    print()
    print(DIV)
    print(title)
    print(DIV)


def _step(n: int, title: str) -> None:
    print()
    print(DIV2)
    print(f"Step {n} -- {title}")
    print(DIV2)


def run() -> None:
    _banner("Document Schema Inference")

    # ── Check for PDFs ────────────────────────────────────────────────────
    missing = [p for p in PDFS if not p.exists()]
    if missing:
        print()
        print("  Missing sample PDFs. Generate them first:")
        print("    python3 data/_generate_schema_demo_pdfs.py")
        for p in missing:
            print(f"    missing: {p.name}")
        sys.exit(1)

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Four PDFs -- contract, invoice, research paper, policy.")
    print("  Layer 1: KeywordClassifier  (always, no API cost)")
    if has_openai:
        print("  Layer 2: LLMFieldExtractor (OPENAI_API_KEY detected -- field extraction enabled)")
    else:
        print("  Layer 2: LLMFieldExtractor (no OPENAI_API_KEY -- field extraction skipped)")
        print("           Set OPENAI_API_KEY to enable typed field extraction.")

    # ── Step 1: Build pipeline ────────────────────────────────────────────
    _step(1, "Build pipeline with two-layer document classification and field extraction")
    print()

    field_extractor = LLMFieldExtractor() if has_openai else None

    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=150,
        overlap=25,
        collection="mixed_docs",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    pipeline.ensure_index("mixed_docs")

    print("  pipeline = DocumentPipeline(")
    print("      classifier=KeywordClassifier(),")
    if has_openai:
        print("      field_extractor=LLMFieldExtractor(),  # one API call per document")
    else:
        print("      # field_extractor not set -- add LLMFieldExtractor() with OPENAI_API_KEY")
    print("      ...,")
    print("  )")

    # ── Step 2: Ingest PDFs ───────────────────────────────────────────────
    _step(2, "Ingest four PDFs")
    print()
    print("  pipeline.ingest(path)  -- PDF parser extracts text page by page,")
    print("  then KeywordClassifier runs once on the full text before chunking.")
    print()

    ingested: dict[str, list[dict]] = {}
    for pdf_path in PDFS:
        print(f"  Ingesting {pdf_path.name} ...", end="", flush=True)
        records = pipeline.ingest(str(pdf_path), collection="mixed_docs")
        ingested[pdf_path.name] = records
        doc_type   = records[0].get("document_type", "unknown") if records else "?"
        n_pages    = len({r.get("page_start") for r in records if r.get("page_start")})
        print(f"  {len(records)} chunks  {n_pages} pages  type={doc_type}")

    print()
    print(f"  Total chunks in store: {store.count('mixed_docs')}")

    # ── Step 3: Classification results ───────────────────────────────────
    _step(3, "KeywordClassifier results")
    print()
    print("  Scored the first 500 words of each PDF against signal phrase sets.")
    print("  Confidence = fraction of signal phrases matched.")
    print()
    print(f"  {'PDF':<28}  {'document_type':<18}  confidence  chunks")
    print(f"  {'-'*28}  {'-'*18}  {'-'*10}  {'-'*6}")

    for pdf_name, records in ingested.items():
        if not records:
            continue
        doc_id   = records[0]["document_id"]
        schema   = pipeline.get_record(doc_id)
        doc_type = schema["document_type"] if schema else "?"
        conf     = f"{schema['confidence']:.4f}" if schema else "?"
        print(f"  {pdf_name:<28}  {doc_type:<18}  {conf:<10}  {len(records)}")

    # ── Step 4: document_type on chunks ──────────────────────────────────
    _step(4, "document_type on every chunk")
    print()
    print("  Every chunk produced from the PDF carries the classified type.")
    print("  Accessible at two locations:")
    print()

    sample_record = ingested["sample_contract.pdf"][0]
    print(f"  chunk['document_type']             = {sample_record.get('document_type')!r}")
    print(f"  chunk['metadata']['document_type'] = {sample_record['metadata'].get('document_type')!r}")
    print(f"  chunk['document_id']               = {sample_record['document_id']!r}")
    print(f"  chunk['page_start']                = {sample_record.get('page_start')!r}")
    print()
    print("  All chunks from one PDF share the same document_type:")
    print()
    for pdf_name, records in ingested.items():
        types = {r.get("document_type") for r in records}
        print(f"  {pdf_name:<28}  {len(records)} chunks  -> document_type={types}")

    # ── Step 5: Schema per document ───────────────────────────────────────
    _step(5, "Schema record per document")
    print()
    print("  pipeline.get_record(document_id) retrieves the schema stored at")
    print("  ingest time. No re-read of the PDF needed.")
    print()

    for pdf_name, records in ingested.items():
        if not records:
            continue
        doc_id = records[0]["document_id"]
        schema = pipeline.get_record(doc_id)
        if not schema:
            continue

        print(f"  [{pdf_name}]")
        print(f"    document_id   : {schema['document_id']}")
        print(f"    document_type : {schema['document_type']}")
        print(f"    confidence    : {schema['confidence']:.4f}  "
              f"(classifier={schema['classifier']})")

        if schema["fields"]:
            print(f"    fields extracted by LLMFieldExtractor:")
            for k, v in schema["fields"].items():
                v_str = str(v)
                if isinstance(v, list):
                    v_str = ", ".join(str(x) for x in v[:3])
                    if len(v) > 3:
                        v_str += f"  (+{len(v)-3} more)"
                display = v_str[:65] + ("..." if len(v_str) > 65 else "")
                print(f"      {k:<26} {display}")
        else:
            print("    fields: (set OPENAI_API_KEY to enable LLM field extraction)")
        print()

    # ── Step 6: Type-aware search results ────────────────────────────────
    _step(6, "document_type in search results")
    print()
    print("  Searching across all four PDFs. Each result carries document_type")
    print("  so the UI can group, badge, or filter results by document category.")
    print()

    results = pipeline.search("termination obligations liability", k=5, collection="mixed_docs")
    print(f"  Query: 'termination obligations liability'  ->  {len(results)} result(s)")
    print()
    print(f"  {'document_type':<18}  {'file':<28}  pg  text preview")
    print(f"  {'-'*18}  {'-'*28}  --  {'-'*30}")
    for r in results:
        doc_type  = r.get("document_type", "(none)")
        file_name = r.get("file_name", "?")
        page      = r.get("page_start", "-")
        snippet   = r.get("text", "").replace("\n", " ")[:45]
        print(f"  {doc_type:<18}  {file_name:<28}  {str(page):<2}  {snippet}...")

    # ── Step 7: DOCUMENT.FIND — filter by document_type ──────────────────
    _step(7, "DOCUMENT.FIND — filter documents by type")
    print()
    print("  pipeline.find_records() queries across all ingested documents.")
    print("  Filter by document_type to narrow instantly — no re-read needed.")
    print()

    for doc_type in ["contract", "invoice", "research_paper", "policy"]:
        matches = pipeline.find_records({"document_type": doc_type})
        names   = [ingested_name
                   for ingested_name, recs in ingested.items()
                   if recs and pipeline.get_record(recs[0]["document_id"]) and
                   pipeline.get_record(recs[0]["document_id"])["document_type"] == doc_type]
        print(f"  find_records({{'document_type': {doc_type!r}}})  ->  {len(matches)} match(es)  {names}")

    # ── Step 8: DOCUMENT.FIND — filter by extracted field ────────────────
    _step(8, "DOCUMENT.FIND — filter by extracted field values")
    print()

    if has_openai:
        print("  Filter on any field extracted by LLMFieldExtractor:")
        print()

        # governing_law
        results = pipeline.find_records({"fields.governing_law": {"contains": "Delaware"}})
        print(f"  find_records({{'fields.governing_law': {{'contains': 'Delaware'}}}}):")
        for r in results:
            print(f"    -> {r['document_type']:<18}  governing_law={r['fields'].get('governing_law')!r}")

        print()

        # total_due > $30,000
        results = pipeline.find_records({"fields.total_due": {"gt": 30000}})
        print(f"  find_records({{'fields.total_due': {{'gt': 30000}}}}):")
        for r in results:
            print(f"    -> {r['document_type']:<18}  total_due={r['fields'].get('total_due')!r}")

        print()

        # multiple conditions: contract + confidence >= 0.7
        results = pipeline.find_records({"document_type": "contract", "confidence": {"gte": 0.7}})
        print(f"  find_records({{'document_type': 'contract', 'confidence': {{'gte': 0.7}}}}):")
        for r in results:
            print(f"    -> {r['document_type']:<18}  confidence={r['confidence']:.4f}")
    else:
        print("  (Set OPENAI_API_KEY to see field-level filtering — fields are empty without LLM extraction)")

    # ── Step 9: DOCUMENT.FIND + DOCUMENT.SEARCH combined ─────────────────
    _step(9, "Two-step query: DOCUMENT.FIND then DOCUMENT.SEARCH")
    print()
    print("  The power pattern: structured filter → targeted text retrieval.")
    print("  Step 1: find matching documents by field value (deterministic).")
    print("  Step 2: retrieve specific text only within those documents.")
    print()

    if has_openai:
        print("  Step 1 — find contract documents:")
        contracts = pipeline.find_records({"document_type": "contract"})
        contract_ids = [r["document_id"] for r in contracts]
        print(f"    find_records({{'document_type': 'contract'}})  ->  {len(contracts)} document(s)")
        for r in contracts:
            print(f"    document_id={r['document_id']}  governing_law={r['fields'].get('governing_law')!r}")

        print()
        print("  Step 2 — search for termination clause within contracts only:")
        results = pipeline.search(
            "termination obligations liability",
            k=3,
            collection="mixed_docs",
            filters={"document_id": contract_ids[0]} if contract_ids else None,
            mode="bm25",
        )
        print(f"    search('termination obligations liability', filter=contract_id)  ->  {len(results)} chunk(s)")
        for r in results:
            snippet = r.get("text", "").replace("\n", " ")[:60]
            print(f"    pg={r.get('page_start', '-')}  section={r.get('section_title', '(none)')!r}")
            print(f"    {snippet}...")
        print()
        print("  Compare: unfiltered search across all 4 PDFs returns policy/invoice")
        print("  chunks for this query. Filtered search goes directly to the right document.")
    else:
        print("  (Showing document_type-only filter — set OPENAI_API_KEY for field-level filters)")
        print()
        contracts = pipeline.find_records({"document_type": "contract"})
        contract_ids = [r["document_id"] for r in contracts]
        print(f"  Step 1 — find contracts: {len(contracts)} document(s) matched")
        print()
        print("  Step 2 — search within contracts only:")
        if contract_ids:
            results = pipeline.search(
                "termination obligations liability",
                k=3,
                collection="mixed_docs",
                filters={"document_id": contract_ids[0]},
                mode="bm25",
            )
            print(f"    {len(results)} chunk(s) from contract document only")
            for r in results:
                snippet = r.get("text", "").replace("\n", " ")[:60]
                print(f"    pg={r.get('page_start', '-')}  {snippet}...")

    # ── Step 10: DOCUMENT.GET_RECORD op ──────────────────────────────────
    _step(10, "DOCUMENT.GET_RECORD via DocumentDataSource")
    print()
    print("  When using DocumentDataSource (MCP / registry / API), the same")
    print("  schema is available as a structured op:")
    print()
    print("    schema = source.execute_query({")
    print("        'op': 'DOCUMENT.GET_RECORD',")
    print("        'document_id': document_id,")
    print("    })")
    print()
    invoice_doc_id = ingested["sample_invoice.pdf"][0]["document_id"]
    invoice_schema = pipeline.get_record(invoice_doc_id)
    print("  Example response for sample_invoice.pdf:")
    print(f"    document_type : {invoice_schema['document_type']}")
    print(f"    confidence    : {invoice_schema['confidence']:.4f}")
    print(f"    chunk_id      : {invoice_schema['chunk_id']}")
    if invoice_schema["fields"]:
        print("    fields:")
        for k, v in invoice_schema["fields"].items():
            print(f"      {k:<26} {v}")
    else:
        print("    fields: {}  (add LLMFieldExtractor to populate)")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  Two-layer document classification and field extraction + structured query:")
    print()
    print("    Layer 1 -- type detection (no API cost, no extra deps):")
    print("      classifier=KeywordClassifier()")
    print("      -> chunk['document_type']  and  schema['confidence']")
    print()
    print("    Layer 2 -- field extraction (one API call per document):")
    print("      field_extractor=LLMFieldExtractor()  # needs OPENAI_API_KEY")
    print("      -> schema['fields']  e.g. parties, effective_date, total_due, ...")
    print()
    print("    DOCUMENT.FIND -- query across all ingested documents:")
    print("      pipeline.find_records({'document_type': 'contract'})")
    print("      pipeline.find_records({'fields.governing_law': {'contains': 'Delaware'}})")
    print("      pipeline.find_records({'fields.total_due': {'gt': 30000}})")
    print()
    print("    Two-step pattern (structured filter + targeted text retrieval):")
    print("      docs    = pipeline.find_records({'document_type': 'contract'})")
    print("      doc_ids = [d['document_id'] for d in docs]")
    print("      chunks  = pipeline.search(query, filters={'document_id': doc_ids[0]})")
    print()
    print("  Supported document types (KeywordClassifier):")
    for t in ["contract", "invoice", "policy", "research_paper", "financial_report", "unknown"]:
        print(f"    * {t}")
    print()
    if not has_openai:
        print("  To enable LLM field extraction and field-level filtering:")
        print("    OPENAI_API_KEY=sk-... python3 examples/feature-demos/schema_inference_demo.py")
        print()


if __name__ == "__main__":
    run()
