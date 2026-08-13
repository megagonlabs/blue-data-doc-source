"""
Multi-Document Collections
==========================
Shows that a single collection holds many documents, search spans all of them,
and every result carries a full provenance chain back to its source.

Story:
  A company has four documents in one collection — an employee benefits policy,
  a data security policy, an engineering handbook, and a vendor contracts guide.
  An agent searches the collection. Results come from different documents. Each
  result knows exactly which document, which section, and which page it came from.

What this demo covers:
  • Ingesting multiple documents into one collection (one ingest call per file)
  • Collection inventory: N documents, M chunks
  • Targeted queries  — results come from one specific document
  • Cross-document queries — results span multiple documents
  • Provenance on every result: file_name, section_title, page
  • How an agent cites its answer with a source reference

Run:
    python3 examples/scenario-demos/multi_document_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import BasicEnricher, DocumentPipeline, DummyEmbeddingModel

COLLECTION = "company_knowledge"
DIV  = "=" * 72
DIV2 = "-" * 72

# ── Document corpus ───────────────────────────────────────────────────────────
# Four documents, each a distinct topic with section headers so SectionChunker
# detects structure and propagates section_title to every chunk.

DOCUMENTS = {
    "benefits_policy.txt": """\
# Employee Benefits Policy

## Health Insurance
All full-time employees are eligible for health insurance from their first day.
The company covers 80% of the premium for individual coverage and 60% for family
coverage. Employees must enroll within 30 days of their start date or during the
annual open enrollment period in November.

## Dental and Vision
Dental coverage includes two cleanings per year at no cost. Major dental work
carries a 50% co-insurance after the $200 annual deductible is met. Vision
coverage provides one eye exam per year and a $150 annual allowance for frames
or contact lenses.

## Retirement Plan
The company matches 401(k) contributions up to 4% of base salary. Employees are
fully vested after three years of service. The plan offers a range of index funds
and target-date funds managed by Fidelity.

## Parental Leave
Primary caregivers receive 16 weeks of fully paid parental leave. Secondary
caregivers receive 6 weeks. Leave may be taken any time within the first year
after birth, adoption, or foster placement.
""",

    "security_policy.txt": """\
# Data Security Policy

## Data Classification
All company data is classified into four tiers: Public, Internal, Confidential,
and Restricted. Restricted data includes personally identifiable information,
payment card data, and source code. Handling rules for each tier are published
in the Data Governance Handbook.

## Access Control
Access to Confidential and Restricted data requires manager approval and is
reviewed quarterly. Employees must use multi-factor authentication for all
systems holding Restricted data. Shared credentials are prohibited.

## Incident Response
Security incidents must be reported to security@company.example within one hour
of discovery. The incident response team will classify the severity and initiate
containment within four hours. All incidents are logged and reviewed monthly.

## Encryption Standards
Data at rest must be encrypted using AES-256. Data in transit must use TLS 1.2
or higher. Encryption keys are rotated annually and stored in the company key
management service, not in application code or configuration files.
""",

    "engineering_handbook.txt": """\
# Engineering Handbook

## Code Review Standards
All code changes require at least one approving review before merging to the
main branch. Reviewers must check for correctness, test coverage, security
implications, and adherence to the style guide. Pull requests must include a
description of the change and a test plan.

## Deployment Process
Deployments follow a three-stage pipeline: staging, canary, and production.
Canary releases expose the change to 5% of production traffic for 30 minutes
before a full rollout. Automated rollback triggers if error rate exceeds 1%
or p99 latency increases by more than 200ms.

## On-Call Responsibilities
On-call engineers are responsible for responding to alerts within 15 minutes
during business hours and 30 minutes after hours. Every incident requires a
post-mortem within 48 hours. The on-call rotation is weekly and uses PagerDuty.

## Testing Requirements
Unit test coverage must remain above 80% for all production services. Integration
tests run on every pull request. Load tests run weekly against the staging
environment. Flaky tests must be fixed or quarantined within two business days.
""",

    "vendor_contracts.txt": """\
# Vendor Contracts Guide

## Procurement Policy
All vendor agreements above $10,000 require legal review and CFO approval.
Agreements above $100,000 require board approval. Procurement requests must be
submitted at least 30 days before the intended start date to allow sufficient
review time.

## Contract Terms
Standard payment terms are net-30 from invoice date. Contracts must include a
termination-for-convenience clause with at least 30 days notice. Liability
caps must be negotiated to a minimum of 12 months of fees paid.

## Vendor Assessment
All vendors handling company data must complete a security questionnaire and
provide their SOC 2 Type II report. Cloud vendors must disclose their sub-processors.
Vendor assessments are renewed annually.

## SLA Requirements
Critical vendors must guarantee 99.9% uptime. SLA breaches entitle the company
to service credits of 10% of monthly fees per incident. Repeated breaches allow
early contract termination without penalty.
""",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def provenance(r: dict) -> str:
    parts = [r.get("file_name", "?")]
    if r.get("section_title"):
        parts.append(f"§ {r['section_title']}")
    if r.get("page_start") is not None:
        parts.append(f"p.{r['page_start']}")
    return " › ".join(parts)


def keyword_search(
    all_chunks: list[dict],
    term: str,
    k: int = 4,
) -> list[dict]:
    hits = [r for r in all_chunks if term.lower() in r.get("text", "").lower()]
    return hits[:k]


def print_results(hits: list[dict], query: str) -> None:
    print(f"\n  Query: \"{query}\"  →  {len(hits)} result(s)\n")
    if not hits:
        print("    (no matches)")
        return
    for i, r in enumerate(hits, 1):
        prov    = provenance(r)
        snippet = r.get("text", "").replace("\n", " ")[:100]
        print(f"  [{i}] {prov}")
        print(f"      \"{snippet}…\"")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    print()
    print(DIV)
    print("Multi-Document Collections")
    print(DIV)
    print()
    print("  One collection. Four documents. Every result knows its source.")
    print()

    # ── Build pipeline ────────────────────────────────────────────────────
    store    = LocalMemoryVectorStore()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=store,
        enricher=BasicEnricher(),
        chunk_size=120,
        overlap=20,
        collection=COLLECTION,
    )
    pipeline.ensure_index(COLLECTION)

    # ── Ingest all four documents ─────────────────────────────────────────
    print(DIV2)
    print("Step 1 — Ingest four documents into one collection")
    print(DIV2)
    print()

    ingested: dict[str, list[dict]] = {}
    for file_name, text in DOCUMENTS.items():
        records = pipeline.ingest_text(
            text=text,
            file_name=file_name,
            mime_type="text/plain",
        )
        ingested[file_name] = records
        sections = {r.get("section_title") for r in records if r.get("section_title")}
        print(f"  {file_name:<30}  {len(records):>3} chunks   "
              f"{len(sections)} sections")

    all_chunks = [r for records in ingested.values() for r in records]
    total_chunks = store.count(COLLECTION)

    print()
    print(f"  Collection: '{COLLECTION}'")
    print(f"  Documents : {len(ingested)}")
    print(f"  Total chunks in store: {total_chunks}")

    # ── Collection inventory ──────────────────────────────────────────────
    print()
    print(DIV2)
    print("Step 2 — Collection inventory")
    print(DIV2)
    print()
    print(f"  {'Document':<30}  {'document_id':<20}  {'Chunks':>6}  Sections")
    print(f"  {'-'*30}  {'-'*20}  {'-'*6}  {'-'*35}")
    for file_name, records in ingested.items():
        if not records:
            continue
        doc_id   = records[0]["document_id"][:20]
        sections = sorted({r.get("section_title") for r in records
                           if r.get("section_title")})
        print(f"  {file_name:<30}  {doc_id:<20}  {len(records):>6}  "
              f"{', '.join(sections[:3])}")
    print()
    print("  Each document has a unique document_id (SHA-256 of content).")
    print("  All chunks share the same collection and are searched together.")

    # ── Targeted queries ──────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Step 3 — Targeted queries: one document answers each")
    print(DIV2)

    targeted = [
        ("401(k)",     "retirement / benefits question"),
        ("AES-256",    "security / encryption question"),
        ("canary",     "engineering / deployment question"),
        ("net-30",     "procurement / contracts question"),
    ]

    for term, label in targeted:
        hits = keyword_search(all_chunks, term)
        docs_hit = {r.get("file_name") for r in hits}
        print(f"\n  [{label}]")
        print_results(hits, term)
        print(f"  Results from: {sorted(docs_hit)}")

    # ── Cross-document queries ────────────────────────────────────────────
    print()
    print(DIV2)
    print("Step 4 — Cross-document queries: multiple documents answer together")
    print(DIV2)
    print()
    print("  Some questions span documents. A search across the collection")
    print("  surfaces relevant chunks from whichever document holds the answer.")
    print()

    cross = [
        ("30 days",   "benefits (enrollment window) + vendor contracts (notice period, procurement lead time)"),
        ("annually",  "security (key rotation, access review) + benefits (vision allowance) + vendor (assessment renewal)"),
        ("approval",  "security (access control) + vendor contracts (procurement policy)"),
    ]

    for term, note in cross:
        hits  = keyword_search(all_chunks, term, k=6)
        by_doc: dict[str, list[dict]] = {}
        for r in hits:
            by_doc.setdefault(r.get("file_name", "?"), []).append(r)

        print(f"  Query: \"{term}\"")
        print(f"  Note : {note}")
        print(f"  Hits : {len(hits)} across {len(by_doc)} document(s)")
        for doc, doc_hits in sorted(by_doc.items()):
            sections = [r.get("section_title") or "(no section)" for r in doc_hits]
            print(f"    • {doc}  →  {len(doc_hits)} chunk(s)  "
                  f"[{', '.join(dict.fromkeys(sections))}]")
        print()

    # ── Provenance citation ───────────────────────────────────────────────
    print(DIV2)
    print("Step 5 — Provenance citation: how an agent grounds its answer")
    print(DIV2)
    print()
    print("  Question: What must a vendor provide before handling company data?")
    print()

    hits = keyword_search(all_chunks, "security questionnaire SOC 2", k=1)
    if not hits:
        hits = keyword_search(all_chunks, "security questionnaire", k=1)
    if hits:
        r       = hits[0]
        snippet = r.get("text", "").replace("\n", " ").strip()
        print("  Answer (grounded):")
        print(f"    \"{snippet}\"")
        print()
        print(f"  Source     : {provenance(r)}")
        print(f"  document_id: {r.get('document_id')}")
        print(f"  chunk_id   : {r.get('chunk_id')}")
        print()
        print("  The agent can cite exactly which document and section this came")
        print("  from — without any post-hoc metadata lookup.")

    # ── Independent deletion ──────────────────────────────────────────────
    print()
    print(DIV2)
    print("Step 6 — Independent deletion: remove one document, others are untouched")
    print(DIV2)
    print()

    target_file = "vendor_contracts.txt"
    target_records = ingested[target_file]
    target_doc_id  = target_records[0]["document_id"]
    before = store.count(COLLECTION)

    pipeline.delete(
        document_id=target_doc_id,
        chunk_count=len(target_records),
        collection=COLLECTION,
    )
    after = store.count(COLLECTION)

    print(f"  Deleted: {target_file}  ({len(target_records)} chunks)")
    print(f"  Store before: {before}  →  after: {after}  "
          f"(removed {before - after})")
    print()

    remaining_files = {
        r.get("file_name") for r in store._store.get(COLLECTION, {}).values()
    }
    print(f"  Remaining documents: {sorted(remaining_files)}")
    print()
    vendor_gone = not any(
        r.get("file_name") == target_file
        for r in store._store.get(COLLECTION, {}).values()
    )
    others_intact = len(remaining_files) == len(DOCUMENTS) - 1
    print(f"  Vendor contracts removed : {'✓' if vendor_gone else '✗'}")
    print(f"  Other documents intact   : {'✓' if others_intact else '✗'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(DIV)
    print("Summary")
    print(DIV)
    print()
    print("  One collection, many documents:")
    print("    pipeline.ingest_text(text, file_name=...)  ← one call per document")
    print("    pipeline.ingest('file.pdf', collection=...) ← or directly from file")
    print()
    print("  Search spans the entire collection:")
    print("    pipeline.search('query', k=5)  ← returns top chunks across all docs")
    print()
    print("  Every result carries full provenance:")
    print("    result['file_name']     → which document")
    print("    result['section_title'] → which section")
    print("    result['page_start']    → which page (PDF/page-aware ingestion)")
    print("    result['document_id']   → stable content-addressed ID")
    print()
    print("  Each document is independently deletable:")
    print("    pipeline.delete(document_id=..., chunk_count=...)")
    print("    → removes that document's chunks; others are untouched")
    print()
    print("  Note: this demo uses DummyEmbeddingModel (SHA-256 hash vectors).")
    print("  For real semantic search across documents, use OpenAIEmbeddingModel.")
    print("  See examples/backends/openai_embeddings_demo.py")
    print()


if __name__ == "__main__":
    run()
