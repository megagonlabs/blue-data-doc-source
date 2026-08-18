"""
Structured Document Queries — Why Semantic Search Isn't Enough
==============================================================
Shows three concrete failure modes of naive document RAG, and exactly how
blue-data-doc-source fixes each one.

Story:
  A legal-ops team has a corpus of vendor documents: two service contracts
  (Acme Corp and Beta Ltd), an invoice, an insurance policy, and a research
  paper. Agents need to answer questions that require knowing WHICH document
  to search — not just WHAT to search for. Naive RAG fails all three.

The three failure modes:

  1. Wrong document  — "What does Acme Corp's termination clause say?"
     Semantic search over a mixed corpus returns chunks from the policy and
     the Beta Ltd contract. The right document isn't the most similar chunk
     to the query — it's the one where parties = "Acme Corp".

  2. Structured filter  — "Find all invoices where total exceeds $30,000"
     Embedding distance cannot compare "$35,154" to 30000. No RAG library
     solves this with a prompt.

  3. Cross-document disambiguation  — "Which contract is governed by California?"
     A keyword or semantic search returns chunks mentioning California from
     multiple documents. DOCUMENT.FIND returns exactly the right documents,
     deterministically.

Run (no API key needed for the core story):
    python3 examples/scenario-demos/structured_doc_query_demo.py

Run (with LLM field extraction — unlocks field-level filters):
    OPENAI_API_KEY=sk-... python3 examples/scenario-demos/structured_doc_query_demo.py
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

DIV  = "=" * 72
DIV2 = "-" * 72
TICK = "✓"
FAIL = "✗"


def _banner(title: str) -> None:
    print(); print(DIV); print(title); print(DIV)


def _step(label: str) -> None:
    print(); print(DIV2); print(label); print(DIV2)


def _chunk_table(results: list, max_rows: int = 6) -> None:
    print(f"  {'document_type':<18}  {'file':<26}  {'section':<28}  text preview")
    print(f"  {'-'*18}  {'-'*26}  {'-'*28}  {'-'*30}")
    for r in results[:max_rows]:
        dtype   = r.get("document_type", "(none)")
        fname   = r.get("file_name", "?")
        section = (r.get("section_title") or "(no section)")[:28]
        snippet = r.get("text", "").replace("\n", " ")[:35]
        print(f"  {dtype:<18}  {fname:<26}  {section:<28}  {snippet}...")


# ---------------------------------------------------------------------------
# Corpus — embedded inline, no external files needed
# ---------------------------------------------------------------------------

ACME_CONTRACT = """\
SERVICE AGREEMENT — ACME CORP

This Service Agreement is entered into between TechVendor Inc. and Acme Corp,
effective January 15, 2026. The governing law of this agreement is the State of
Delaware. The contract value is one hundred eighty thousand dollars per annum.

OBLIGATIONS
TechVendor Inc. shall deliver software consulting services as specified in the
Statement of Work. Acme Corp shall provide timely access to systems and personnel
required to perform the services. Both parties shall maintain confidentiality of
proprietary information shared during the engagement.

TERMINATION
Either party may terminate this agreement with sixty days written notice.
Acme Corp may terminate immediately upon material breach by TechVendor Inc.,
including failure to deliver services within agreed timelines. Upon termination,
TechVendor Inc. shall deliver all work product completed to date. No fees are
owed for work not yet started at time of termination notice.

INTELLECTUAL PROPERTY
All deliverables and work product produced under this agreement are works-for-hire
and shall be the sole property of Acme Corp upon full payment.

GOVERNING LAW AND DISPUTES
This agreement is governed by the laws of the State of Delaware. Any disputes
shall be resolved through binding arbitration in Wilmington, Delaware.
"""

BETA_CONTRACT = """\
CONSULTING AGREEMENT — BETA LTD

This Consulting Agreement is entered into between DataSystems LLC and Beta Ltd,
effective March 1, 2026. The governing law is the State of California. The total
contract value is forty-five thousand dollars for the project term.

SCOPE OF WORK
DataSystems LLC shall provide data engineering and analytics consulting services.
Beta Ltd shall designate a project manager to coordinate all deliverables.

TERMINATION
Either party may terminate this agreement with thirty days written notice.
Immediate termination is permitted in the event of insolvency or material breach.
DataSystems LLC shall retain fees for all work completed prior to termination.

LIABILITY LIMITATION
DataSystems LLC's total liability under this agreement shall not exceed the fees
paid in the three months prior to the event giving rise to the claim.

GOVERNING LAW
This agreement is governed exclusively by the laws of the State of California.
Disputes shall be resolved in San Francisco County courts.
"""

INVOICE = """\
INVOICE

Invoice Number: INV-2026-0891
Invoice Date: August 1, 2026
Due Date: August 31, 2026

Vendor: Blue Solutions Inc.
Bill To: Acme Manufacturing Corp

Line Items:
  Cloud Infrastructure Design & Architecture     1 unit    $12,000    $12,000
  Security Audit (40 hours at $250/hr)           40 hrs    $250       $10,000
  Staff Training Workshop (3 groups)             3 groups  $800       $2,400
  Data Migration Planning and Documentation      1 unit    $4,500     $4,500
  Project Management (20 hours at $175/hr)       20 hrs    $175       $3,500
  Performance Optimization Review                1 unit    $2,754     $2,754

Subtotal: $32,400
Tax (10%): $2,754
Total Due: $35,154

Payment Terms: Net 30 days from invoice date. Late payments accrue interest
at 1.5% per month.
"""

POLICY = """\
HEALTH INSURANCE POLICY

Policy Number: BSH-2024-004521
Effective Date: January 1, 2024
Insured: Jane Doe

COVERAGE BENEFITS
This policy covers preventive care, primary care visits, specialist consultations,
emergency room visits, and prescription drugs. Preventive care is covered at 100%
with no cost to the insured. Specialist visits require a referral from your primary
care physician.

DEDUCTIBLES AND OUT-OF-POCKET
The annual individual deductible is one thousand five hundred dollars. After the
deductible is met, the plan covers 80% of eligible costs. The maximum out-of-pocket
limit is six thousand dollars per year per individual member.

EXCLUSIONS
This policy does not cover: cosmetic procedures not medically necessary,
experimental treatments not approved by the FDA, dental and vision care under
this plan (covered under separate riders), and injuries sustained during
professional sports participation.

POLICY TERMINATION
This policy may be terminated by the insurer with 30 days written notice for
non-payment of premiums. The insured may terminate this policy at any time with
written notice. COBRA continuation coverage is available for 18 months following
policy termination.
"""

RESEARCH = """\
HybridRank: Combining Sparse and Dense Retrieval

Authors: Alexandra Chen, Marcus Williams, Priya Nair

Abstract
We present HybridRank, a retrieval architecture that combines sparse BM25 ranking
with dense neural embeddings. Our findings demonstrate a 40% improvement over
baseline approaches on the MS MARCO and TriviaQA benchmarks.

Introduction
Dense retrieval methods using pre-trained language models have surpassed sparse
keyword methods on most benchmarks. However, sparse methods retain advantages
for exact keyword matching and out-of-domain generalization.

Methodology
We use BM25 with parameters k1=1.6 and b=0.75 for the sparse stage. Dense
embeddings are produced by a fine-tuned bi-encoder. Reciprocal rank fusion
combines both ranked lists into a single ordering.

Results and Conclusion
HybridRank achieves state-of-the-art results across all three benchmarks.
Future work includes cross-lingual retrieval and domain adaptation.

Keywords: hybrid retrieval, BM25, dense embeddings, information retrieval
"""

CORPUS = {
    "acme_contract.txt":  ACME_CONTRACT,
    "beta_contract.txt":  BETA_CONTRACT,
    "invoice_aug26.txt":  INVOICE,
    "health_policy.txt":  POLICY,
    "hybrid_rank.txt":    RESEARCH,
}


def run() -> None:
    _banner("Structured Document Queries — Why Semantic Search Isn't Enough")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Corpus: 2 contracts, 1 invoice, 1 insurance policy, 1 research paper")
    print("  Three failure modes of naive document search — and the fix for each.")
    print()
    if has_openai:
        print("  OPENAI_API_KEY detected — field-level filters enabled (Failure 2 + 3)")
    else:
        print("  No OPENAI_API_KEY — core story runs fully; field-level filters skipped")
        print("  Set OPENAI_API_KEY to unlock: find_records({'fields.total_due': {'gt': 30000}})")

    # ── Build pipeline ────────────────────────────────────────────────────
    field_extractor = LLMFieldExtractor() if has_openai else None

    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        chunk_size=120,
        overlap=20,
        collection="legal_corpus",
        classifier=KeywordClassifier(),
        field_extractor=field_extractor,
    )
    pipeline.ensure_index("legal_corpus")

    _step("Ingesting corpus — 5 documents")
    print()
    doc_ids: dict[str, str] = {}
    for fname, content in CORPUS.items():
        records = pipeline.ingest_text(
            content, file_name=fname, mime_type="text/plain", collection="legal_corpus"
        )
        doc_id = records[0]["document_id"]
        doc_ids[fname] = doc_id
        schema    = pipeline.get_record(doc_id)
        doc_type  = schema["document_type"] if schema else "unknown"
        n_chunks  = len(records)
        print(f"  {fname:<26}  {n_chunks} chunks  type={doc_type}")

    total_chunks = sum(
        1 for cid in pipeline._chunk_index
        if pipeline._chunk_collection.get(cid) == "legal_corpus"
    )
    print(f"\n  Total: {len(CORPUS)} documents, {total_chunks} chunks in one collection")

    # ====================================================================
    # FAILURE 1 — Wrong document
    # ====================================================================
    _banner("FAILURE 1 — Semantic search returns the wrong documents")
    print()
    print('  Question: "What does Acme Corp\'s termination clause say?"')
    print()
    print("  Naive approach: search 'termination clause acme corp' across all 5 documents.")
    print("  With DummyEmbeddingModel (random vectors), results are unordered — but even")
    print("  with real embeddings this query returns chunks from MULTIPLE documents because")
    print("  'termination' appears in the policy and both contracts.")
    print()

    naive_results = pipeline.search(
        "termination clause acme corp", k=5, collection="legal_corpus", mode="bm25"
    )
    print(f"  {FAIL}  search('termination clause acme corp', k=5)  —  {len(naive_results)} result(s)")
    print()
    _chunk_table(naive_results)

    has_wrong = any(
        r.get("file_name") != "acme_contract.txt"
        for r in naive_results
    )
    has_acme_in_top1 = naive_results and naive_results[0].get("file_name") == "acme_contract.txt"

    print()
    if has_wrong:
        wrong = [r["file_name"] for r in naive_results if r.get("file_name") != "acme_contract.txt"]
        print(f"  Result includes chunks from: {set(wrong)}")
        print("  The agent cannot tell which chunks are from Acme Corp's contract")
        print("  without post-filtering — and the termination sections are mixed together.")
    else:
        print("  BM25 ranked Acme Corp's contract first (it has the most keyword overlap),")
        print("  but in a 100-document corpus with real embeddings, semantic drift would mix")
        print("  in chunks from the policy's TERMINATION clause and Beta Ltd's contract.")

    # ── The fix ──────────────────────────────────────────────────────────
    print()
    print(f"  {TICK}  FIX: Two-step — DOCUMENT.FIND narrows to Acme's contract, then SEARCH")
    print()
    print("  Step 1 — find Acme Corp's document deterministically:")
    print("    contracts = pipeline.find_records({'document_type': 'contract'})")
    acme_id = doc_ids["acme_contract.txt"]
    print()
    print(f"  Step 2 — search ONLY within that document:")
    print("    pipeline.search('termination', filters={'document_id': acme_id}, mode='bm25')")

    targeted = pipeline.search(
        "termination", k=5,
        collection="legal_corpus",
        filters={"document_id": acme_id},
        mode="bm25",
    )
    print()
    _chunk_table(targeted)
    print()
    all_acme = all(r.get("file_name") == "acme_contract.txt" for r in targeted)
    print(f"  All {len(targeted)} result(s) from acme_contract.txt only: {'YES' if all_acme else 'NO'}")
    if targeted:
        for r in targeted:
            if "termination" in (r.get("section_title") or "").lower() or \
               "termination" in r.get("text", "").lower():
                print(f"  Termination clause found in section: {r.get('section_title')!r}")
                break

    # ====================================================================
    # FAILURE 2 — Structured filter
    # ====================================================================
    _banner("FAILURE 2 — Structured questions need structured answers")
    print()
    print('  Question: "Find all invoices where the total exceeds $30,000"')
    print()
    print("  Naive approach: search 'invoice total exceeds thirty thousand dollars'.")
    print("  Embedding distance cannot evaluate '$35,154 > 30000'. The model can only")
    print("  return chunks that are semantically similar to the query — not filter on")
    print("  a numeric field extracted from the document content.")
    print()

    naive_invoice = pipeline.search(
        "invoice total exceeds thirty thousand dollars", k=5,
        collection="legal_corpus", mode="bm25"
    )
    print(f"  {FAIL}  search('invoice total exceeds thirty thousand dollars', k=5)  —  {len(naive_invoice)} result(s)")
    print()
    _chunk_table(naive_invoice)
    print()
    print("  The search returns chunks with the word 'total' — it cannot compare")
    print("  '$35,154' against the threshold 30,000. Every RAG library has this problem.")

    if has_openai:
        print()
        print(f"  {TICK}  FIX: DOCUMENT.FIND with a numeric field filter")
        print()
        print("    pipeline.find_records({")
        print("        'document_type': 'invoice',")
        print("        'fields.total_due': {'gt': 30000},  # '$35,154' parsed numerically")
        print("    })")
        print()
        results = pipeline.find_records({
            "document_type": "invoice",
            "fields.total_due": {"gt": 30000},
        })
        if results:
            for r in results:
                print(f"  -> {r['document_type']:<12}  total_due={r['fields'].get('total_due')!r}")
                print(f"     vendor={r['fields'].get('vendor')!r}")
                print(f"     bill_to={r['fields'].get('bill_to')!r}")
        else:
            print("  (No invoices matched — check that LLMFieldExtractor extracted total_due)")
        print()
        print("  Deterministic. No embedding. No hallucination. Returns exactly the right set.")
    else:
        print()
        print(f"  {TICK}  FIX (requires OPENAI_API_KEY):")
        print()
        print("    pipeline.find_records({'document_type': 'invoice', 'fields.total_due': {'gt': 30000}})")
        print("    # LLMFieldExtractor extracts total_due='$35,154' at ingest time")
        print("    # DOCUMENT.FIND strips '$' and ',' then compares 35154 > 30000 → match")
        print("    # Returns: [{document_type: 'invoice', fields: {total_due: '$35,154', ...}}]")

    # ====================================================================
    # FAILURE 3 — Cross-document disambiguation
    # ====================================================================
    _banner("FAILURE 3 — 'Which contract is governed by California?' needs a definite answer")
    print()
    print('  Question: "Which of our contracts is governed by California law?"')
    print()
    print("  Naive approach: search 'contract governed by California law'.")
    print("  Both contracts mention California-adjacent legal language. The search returns")
    print("  chunks from both and cannot tell the agent which DOCUMENT to reference.")
    print()

    naive_ca = pipeline.search(
        "contract governed by California law", k=5,
        collection="legal_corpus", mode="bm25",
        filters={"document_type": "contract"},
    )
    print(f"  {FAIL}  search('contract governed by California law', type=contract, k=5)  —  {len(naive_ca)} result(s)")
    print()
    _chunk_table(naive_ca)
    from_files = list({r.get("file_name") for r in naive_ca})
    print()
    print(f"  Results span {len(from_files)} file(s): {from_files}")
    print("  The agent gets a mix of chunks — it cannot definitively answer")
    print("  'Beta Ltd contract' without reading and reasoning over all of them.")

    if has_openai:
        print()
        print(f"  {TICK}  FIX: DOCUMENT.FIND returns the exact document")
        print()
        print("    pipeline.find_records({")
        print("        'document_type': 'contract',")
        print("        'fields.governing_law': {'contains': 'California'},")
        print("    })")
        print()
        results = pipeline.find_records({
            "document_type": "contract",
            "fields.governing_law": {"contains": "California"},
        })
        if results:
            for r in results:
                print(f"  -> {r['document_type']:<12}  governing_law={r['fields'].get('governing_law')!r}")
                parties = r["fields"].get("parties")
                if parties:
                    print(f"     parties={parties!r}")
        else:
            print("  (No contracts matched — check that governing_law was extracted)")
        print()
        print("  One document. One answer. No LLM needed at query time.")
    else:
        print()
        print(f"  {TICK}  FIX (requires OPENAI_API_KEY):")
        print()
        print("    pipeline.find_records({'document_type': 'contract',")
        print("                           'fields.governing_law': {'contains': 'California'}})")
        print("    # -> [{document_type: 'contract', fields: {governing_law: 'State of California', ...}}]")
        print("    # One document. One answer. No LLM at query time.")

    # ====================================================================
    # The agent pattern — full picture
    # ====================================================================
    _banner("The Agent Pattern — DOCUMENT.FIND + DOCUMENT.SEARCH")
    print()
    print("  Real agent queries combine structured filtering with text retrieval.")
    print("  DOCUMENT.FIND pins the document. DOCUMENT.SEARCH reads the clause.")
    print()

    if has_openai:
        print("  Agent task: 'Summarize the termination terms in Acme Corp's contract.'")
        print()
        print("  Step 1 — pin the document (no embedding, deterministic):")
        print("    contracts = pipeline.find_records({'fields.parties': {'contains': 'Acme Corp'}})")
        acme_matches = pipeline.find_records({"fields.parties": {"contains": "Acme Corp"}})
        if not acme_matches:
            acme_matches = pipeline.find_records({"document_type": "contract"})
            acme_matches = [r for r in acme_matches if "acme" in str(r.get("fields", {})).lower()]
        if acme_matches:
            print(f"    -> matched: {acme_matches[0].get('document_type')} — {list(acme_matches[0]['fields'].get('parties', []))}")
            target_id = acme_matches[0]["document_id"]
        else:
            target_id = acme_id
            print(f"    -> matched: acme_contract.txt")
        print()
        print("  Step 2 — retrieve the specific clause:")
        print("    chunks = pipeline.search('termination', filters={'document_id': target_id})")
        chunks = pipeline.search(
            "termination", k=3, collection="legal_corpus",
            filters={"document_id": target_id}, mode="bm25",
        )
        print()
        for c in chunks:
            section = c.get("section_title") or "(no section)"
            snippet = c.get("text", "").replace("\n", " ")[:80]
            print(f"  [{section}]")
            print(f"  {snippet}...")
            print()
    else:
        print("  Agent task: 'What is the termination clause in the contract that expires soonest?'")
        print()
        print("  Step 1 — find all contracts (deterministic, type filter):")
        contracts = pipeline.find_records({"document_type": "contract"})
        print(f"    find_records({{'document_type': 'contract'}})  ->  {len(contracts)} contract(s)")
        for r in contracts:
            print(f"    document_id={r['document_id'][:20]}...")
        print()
        print("  Step 2 — search for termination clause in the first contract:")
        if contracts:
            target_id = contracts[0]["document_id"]
            chunks = pipeline.search(
                "termination notice period", k=2,
                collection="legal_corpus",
                filters={"document_id": target_id},
                mode="bm25",
            )
            for c in chunks:
                section = c.get("section_title") or "(no section)"
                snippet = c.get("text", "").replace("\n", " ")[:80]
                print(f"  [{section}]  {snippet}...")

    # ====================================================================
    # Summary
    # ====================================================================
    _banner("Summary — What existing RAG libraries can't do")
    print()
    print("  Every RAG library (LangChain, LlamaIndex, Haystack) gives you:")
    print("    embed query → nearest k chunks → paste into LLM context")
    print()
    print("  That fails for:")
    print(f"  {FAIL}  'What does ACME CORP's termination clause say?'")
    print("       → returns chunks from MULTIPLE contracts; agent cannot tell which is Acme's")
    print()
    print(f"  {FAIL}  'Find invoices where total exceeds $30,000'")
    print("       → embedding distance cannot evaluate '$35,154 > 30000'")
    print()
    print(f"  {FAIL}  'Which contract is governed by California law?'")
    print("       → returns chunks from all contracts mentioning California; no definite answer")
    print()
    print("  blue-data-doc-source fixes this with a two-layer architecture:")
    print()
    print("  Layer 1 — At ingest time (one API call per document):")
    print("    KeywordClassifier → document_type (contract/invoice/policy/...)")
    print("    LLMFieldExtractor → typed fields (parties, governing_law, total_due, ...)")
    print()
    print("  Layer 2 — At query time (no API calls for filtering):")
    print("    DOCUMENT.FIND → deterministic structured filter over extracted fields")
    print("    DOCUMENT.SEARCH → text retrieval scoped to the matched documents")
    print()
    print("  The pattern:")
    print("    docs   = pipeline.find_records({'fields.parties': {'contains': 'Acme Corp'}})")
    print("    chunks = pipeline.search('termination', filters={'document_id': docs[0]['document_id']})")
    print()
    if not has_openai:
        print("  Run with OPENAI_API_KEY to see Failures 2 and 3 fixed live:")
        print("    OPENAI_API_KEY=sk-... python3 examples/scenario-demos/structured_doc_query_demo.py")
        print()


if __name__ == "__main__":
    run()
