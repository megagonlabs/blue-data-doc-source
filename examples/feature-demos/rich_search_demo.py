"""Rich Search Demo
==================
Shows DOCUMENT.SEARCH_RICH versus plain DOCUMENT.SEARCH.

Plain search returns a flat list of raw chunk dicts — text, score, metadata.
Rich search wraps the same KNN query and adds three layers on top:

  1. Provenance block — document_id, file_name, chunk_id, page (PDF), section
  2. Neighbor context — text of adjacent chunks (before / after each result)
  3. by_document grouping — results reorganised by source document with
     match_count and deduplicated top_sections

Run:
    python3 examples/feature-demos/rich_search_demo.py

No API key required. DummyEmbeddingModel is used so all deps are local.
(Results are not semantically ranked — use OpenAIEmbeddingModel in production.)
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel

DATA    = Path(__file__).parent.parent.parent / "data"
COLL    = "rich_demo"
DIV     = "=" * 72
DIV2    = "-" * 72

CONTRACT_TEXT = """\
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2024,
between Acme Corp ("Provider") and Blue Holdings Inc ("Client").

1. Scope of Services
The Provider agrees to deliver software development services as described in
Schedule A attached hereto. Services include system design, implementation,
testing, and documentation.

2. Payment Terms
Client shall pay Provider a monthly retainer of $15,000 due on the first
business day of each month. Late payments incur a 1.5% monthly interest charge.

3. Term and Termination
This Agreement commences on January 1, 2024 and continues for twelve (12)
months unless terminated earlier. Either party may terminate with 30 days
written notice. Provider may terminate immediately for non-payment.

4. Confidentiality
Both parties agree to maintain strict confidentiality regarding proprietary
information shared under this Agreement. This obligation survives termination.

5. Governing Law
This Agreement is governed by the laws of the State of California. Disputes
shall be resolved by binding arbitration in San Francisco, California.

6. Warranties
Provider warrants that services will be performed in a professional manner
consistent with industry standards. Provider makes no other warranties,
express or implied, including warranties of merchantability.

7. Indemnification
Each party shall indemnify and hold harmless the other party from any claims,
damages, or losses arising from its own acts or omissions under this Agreement.
"""


def _banner(title: str) -> None:
    print()
    print(DIV)
    print(f"  {title}")
    print(DIV)


def _step(n: int, title: str) -> None:
    print()
    print(DIV2)
    print(f"  Step {n} — {title}")
    print(DIV2)


def _wrap(text: str, width: int = 66, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def run() -> None:
    _banner("Rich Search Demo — DOCUMENT.SEARCH_RICH vs DOCUMENT.SEARCH")

    # ── Step 1: ingest two documents ─────────────────────────────────────
    _step(1, "Ingest two documents")

    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=120,
        overlap=20,
        collection=COLL,
    )
    pipeline.ensure_index(collection=COLL)

    policy_path = DATA / "sample_policy.txt"
    policy_chunks = pipeline.ingest(str(policy_path))

    contract_chunks = pipeline.ingest_text(
        CONTRACT_TEXT,
        file_name="service_agreement.txt",
        mime_type="text/plain",
        collection=COLL,
    )

    print()
    print(f"  Document 1: {policy_path.name}")
    print(f"    chunks: {len(policy_chunks)}  "
          f"document_id: {policy_chunks[0]['document_id']}")
    print()
    print(f"  Document 2: service_agreement.txt")
    print(f"    chunks: {len(contract_chunks)}  "
          f"document_id: {contract_chunks[0]['document_id']}")
    print()
    total = len(policy_chunks) + len(contract_chunks)
    print(f"  Total chunks in store: {total}")

    # ── Step 2: plain DOCUMENT.SEARCH ────────────────────────────────────
    _step(2, "Plain DOCUMENT.SEARCH — flat list of raw chunk dicts")

    plain = pipeline.search("coverage and payment terms", k=4, collection=COLL)

    print()
    print(f"  pipeline.search('coverage and payment terms', k=4)")
    print(f"  Returns: list[dict]  ({len(plain)} results)")
    print()
    for i, r in enumerate(plain):
        score   = r.get("score", 0)
        snippet = (r.get("text") or "")[:80].replace("\n", " ")
        section = r.get("section_title", "")
        print(f"  [{i}] score={score:.4f}  file={r.get('file_name')}  "
              f"section={section!r}")
        print(f"       text: {snippet!r}…")
    print()
    print("  Keys on each dict:", sorted(k for k in plain[0].keys()
                                         if k not in ("embedding",)))

    # ── Step 3: DOCUMENT.SEARCH_RICH full output ─────────────────────────
    _step(3, "DOCUMENT.SEARCH_RICH — full structured response")

    rich = pipeline.search_rich(
        "coverage and payment terms",
        k=4,
        collection=COLL,
        context_chunks=1,
    )

    print()
    print("  pipeline.search_rich('coverage and payment terms', k=4, context_chunks=1)")
    print()
    print("  Top-level keys:")
    for key, val in rich.items():
        if key in ("results", "by_document"):
            print(f"    {key!r}: ({type(val).__name__}, len={len(val)})")
        else:
            print(f"    {key!r}: {val!r}")

    # ── Step 4: provenance anatomy ────────────────────────────────────────
    _step(4, "Provenance block — anatomy of one result")

    print()
    print("  Each result in rich['results'] looks like:")
    print()

    for r in rich["results"]:
        print("  {")
        print(f"    'rank':         {r['rank']},")
        print(f"    'score':        {r['score']:.4f},")
        print(f"    'chunk_id':     {r['chunk_id']!r},")
        snippet = (r.get("text") or "")[:70].replace("\n", " ")
        print(f"    'text':         {snippet!r}…,")
        prov = r["provenance"]
        print( "    'provenance': {")
        print(f"      'document_id': {prov['document_id']!r},")
        print(f"      'file_name':   {prov['file_name']!r},")
        print(f"      'chunk_id':    {prov['chunk_id']!r},")
        if "chunk_index" in prov:
            print(f"      'chunk_index': {prov['chunk_index']},")
        if "page" in prov:
            print(f"      'page':        {prov['page']},")
        if "section" in prov:
            print(f"      'section':     {prov['section']!r},")
        print( "    },")
        ctx = r.get("context", {})
        if ctx:
            before = (ctx.get("before") or "")[:50].replace("\n", " ")
            after  = (ctx.get("after")  or "")[:50].replace("\n", " ")
            print( "    'context': {")
            if "before" in ctx:
                print(f"      'before': {before!r}…,")
            if "after" in ctx:
                print(f"      'after':  {after!r}…,")
            print( "    },")
        else:
            print( "    'context': {}  # no neighbor chunks available")
        print("  }")
        print()

    # ── Step 5: context_chunks parameter ─────────────────────────────────
    _step(5, "context_chunks parameter — neighbor context on/off")

    print()
    print("  context_chunks=1  (default) — includes text from chunks")
    print("  immediately before and after each result in the document.")
    print()

    with_ctx = pipeline.search_rich(
        "termination clause",
        k=3,
        collection=COLL,
        context_chunks=1,
    )
    without_ctx = pipeline.search_rich(
        "termination clause",
        k=3,
        collection=COLL,
        context_chunks=0,
    )

    has_before  = sum(1 for r in with_ctx["results"] if "before" in r.get("context", {}))
    has_after   = sum(1 for r in with_ctx["results"] if "after"  in r.get("context", {}))
    no_ctx_keys = sum(1 for r in without_ctx["results"] if "context" not in r)

    print(f"  context_chunks=1  →  results with 'before': {has_before}, "
          f"results with 'after': {has_after}")
    print(f"  context_chunks=0  →  results with no 'context' key: {no_ctx_keys}")
    print()
    print("  Sample with context (context_chunks=1):")
    for r in with_ctx["results"]:
        ctx = r.get("context", {})
        snippet = (r.get("text") or "")[:60].replace("\n", " ")
        before  = (ctx.get("before") or "")[:50].replace("\n", " ")
        after   = (ctx.get("after")  or "")[:50].replace("\n", " ")
        print(f"    rank={r['rank']}  text: {snippet!r}…")
        if before:
            print(f"           before: {before!r}…")
        if after:
            print(f"           after:  {after!r}…")
    print()
    print("  Caveat: neighbor lookup uses an in-process chunk index populated")
    print("  at ingest time. Chunks ingested in a previous process run are not")
    print("  available as neighbors — 'context' keys are silently omitted for")
    print("  those results. Re-ingest in the same process to restore context.")

    # ── Step 6: by_document grouping ─────────────────────────────────────
    _step(6, "by_document grouping — results organised per source file")

    rich6 = pipeline.search_rich(
        "payment terms coverage agreement",
        k=6,
        collection=COLL,
        context_chunks=0,
    )

    print()
    print("  rich['by_document'] groups results by document_id so you can")
    print("  see which source documents matched and how many chunks each hit.")
    print()
    print(f"  query: 'payment terms coverage agreement'  k=6")
    print(f"  result_count={rich6['result_count']}  "
          f"document_count={rich6['document_count']}")
    print()
    for doc_id, group in rich6["by_document"].items():
        print(f"  Document: {group['file_name']}")
        print(f"    document_id:  {doc_id}")
        print(f"    match_count:  {group['match_count']}")
        if "top_sections" in group:
            print(f"    top_sections: {group['top_sections']}")
        print(f"    matches:")
        for m in group["matches"]:
            snippet = (m.get("text") or "")[:60].replace("\n", " ")
            sec     = m["provenance"].get("section", "")
            print(f"      rank={m['rank']}  score={m['score']:.4f}  "
                  f"§={sec!r}  text: {snippet!r}…")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary — DOCUMENT.SEARCH vs DOCUMENT.SEARCH_RICH")
    print()
    print("  Feature              DOCUMENT.SEARCH        DOCUMENT.SEARCH_RICH")
    print("  " + "-" * 68)
    print("  Return type          list[dict]             dict")
    print("  Ranked chunks        ✓ raw chunk fields     ✓ same KNN query")
    print("  Provenance block     ✗ fields scattered     ✓ {doc_id, file, page, section}")
    print("  Neighbor context     ✗ not available        ✓ context.before / after")
    print("  Grouped by document  ✗ flat list only       ✓ by_document + top_sections")
    print("  Result count / doc   ✗ must count manually  ✓ match_count per document")
    print()
    print("  When to use rich search:")
    print("    - Feeding results to an LLM for answer generation (RAG)")
    print("    - Building provenance-aware UI (show 'page 3, §Coverage' links)")
    print("    - Multi-document corpuses where grouping by source matters")
    print()
    print("  When plain search is enough:")
    print("    - Simple ranked snippet retrieval")
    print("    - When you process results yourself downstream")
    print("    - Testing / debugging — fewer nested fields to inspect")
    print()


if __name__ == "__main__":
    run()
