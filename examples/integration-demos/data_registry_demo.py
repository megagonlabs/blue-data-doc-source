"""
DataRegistry Integration Demo
==============================
Shows DocumentDataSource registered in blue-data's DataRegistry alongside
other source types.  Demonstrates the full registration workflow:

  1. Register DocumentDataSource in a DataRegistry
  2. Ingest documents and classify them
  3. sync_source — registry discovers the ChunkRecord schema from the live source
  4. search_records — search registry metadata (find the source that holds documents)
  5. execute_query — structured DOCUMENT.* ops via the registry
  6. execute_query_nl — natural-language dispatch via the registry

No API key required.  All storage is in-process (MemoryJSONSource,
MemoryVectorSource, DummyEmbeddingModel).

Run:
    python3 examples/integration-demos/data_registry_demo.py
"""
from __future__ import annotations

import hashlib
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data.registry.data_registry import DataRegistry
from blue_data.source.json.memory_json_source import MemoryJSONSource
from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import (
    DocumentDataSource,
    DummyEmbeddingModel,
    KeywordClassifier,
)

DIV  = "=" * 68
DIV2 = "-" * 68

# ---------------------------------------------------------------------------
# Registry embedding — deterministic hash-based vectors (no API key)
# ---------------------------------------------------------------------------

REGISTRY_DIM = 32

def _registry_embed(text: str) -> list[float]:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [math.sin(seed * (i + 1) * 0.1) for i in range(REGISTRY_DIM)]


# ---------------------------------------------------------------------------
# Sample corpus — no files needed
# ---------------------------------------------------------------------------

DOCS = [
    (
        "master_services_agreement.txt",
        """Master Services Agreement
This Agreement is entered into between Acme Corp ("Client") and Blue Solutions Inc ("Vendor").
The parties agree to the following terms governing law California and indemnification provisions.
The agreement covers software development services, consulting, and related deliverables.
Payment terms: net 30 days. Warranty period: 12 months from delivery.
Governing law: State of California. Arbitration clause applies to all disputes.
Either party may terminate this agreement with 30 days written notice.""",
    ),
    (
        "invoice_2026_042.txt",
        """Invoice #2026-042
Bill To: Acme Corp, 100 Main St, San Francisco CA 94102
Vendor: Blue Solutions Inc
Invoice Date: 2026-01-15
Due Date: 2026-02-14
Line Items:
  Software development services   $28,000.00
  Infrastructure setup             $4,500.00
  Training sessions                $2,654.00
Subtotal: $35,154.00
Tax (8.5%): $2,988.09
Amount Due: $38,142.09""",
    ),
    (
        "health_policy_2026.txt",
        """Health Insurance Policy — 2026 Plan Year
Eligibility: All full-time employees working 30+ hours per week.
Coverage: Medical, dental, vision included. Mental health parity applies.
Deductible: $1,500 individual / $3,000 family per plan year.
Out-of-pocket maximum: $5,000 individual / $10,000 family.
Premium contribution: Employee pays 20% of monthly premium.
Exclusions: Cosmetic procedures, experimental treatments.
Open enrollment period: November 1–30 each year.
Qualifying life events allow mid-year changes within 30 days.""",
    ),
]


def main() -> None:
    print()
    print(DIV)
    print("DataRegistry Integration Demo")
    print(DIV)
    print()
    print("  Storage : MemoryJSONSource (registry) + MemoryVectorSource (doc store)")
    print("  Embed   : DummyEmbeddingModel (SHA-256 hash — no API key)")
    print("  Classify: KeywordClassifier (no API key)")
    print()

    # -----------------------------------------------------------------------
    # 1. Build the registry with in-process backends
    # -----------------------------------------------------------------------
    print(DIV2)
    print("Step 1 — Build DataRegistry")
    print(DIV2)

    registry = DataRegistry(
        store=MemoryJSONSource(),
        vector_source=MemoryVectorSource(),
        embedding_fn=_registry_embed,
        vector_dim=REGISTRY_DIM,
        rebuild=True,
        hybrid_config={"vector_threshold": 0.0, "combined_threshold": 0.0},
    )
    print("  DataRegistry created (in-process store)")

    # -----------------------------------------------------------------------
    # 2. Create and register DocumentDataSource
    # -----------------------------------------------------------------------
    print()
    print(DIV2)
    print("Step 2 — Register DocumentDataSource")
    print(DIV2)

    doc_vector_store = MemoryVectorSource()
    doc_source = DocumentDataSource(
        name="legal_corpus",
        embedding_model=DummyEmbeddingModel(),
        vector_source=doc_vector_store,
        classifier=KeywordClassifier(),
        default_collection="legal_corpus",
    )

    registry.register_source(doc_source)
    print("  Registered sources:", registry.list_sources())

    # -----------------------------------------------------------------------
    # 3. Ingest documents
    # -----------------------------------------------------------------------
    print()
    print(DIV2)
    print("Step 3 — Ingest documents")
    print(DIV2)

    doc_source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "legal_corpus"})

    tmp_dir = tempfile.mkdtemp(prefix="blue-doc-demo-")
    for file_name, text in DOCS:
        tmp_path = Path(tmp_dir) / file_name
        tmp_path.write_text(text)
        result = doc_source.execute_query({
            "op":         "DOCUMENT.INGEST",
            "path":       str(tmp_path),
            "collection": "legal_corpus",
        })
        doc_type = result[0].get("document_type", "unknown") if result else "unknown"
        print(f"  {file_name:<40} → {len(result)} chunks  type={doc_type}")

    print()

    # -----------------------------------------------------------------------
    # 4. sync_source — registry discovers ChunkRecord schema from live source
    # -----------------------------------------------------------------------
    print(DIV2)
    print("Step 4 — sync_source (registry discovers schema)")
    print(DIV2)

    registry.sync_source("legal_corpus", recursive=True)
    print("  sync_source complete")

    databases   = registry.list_source_databases("legal_corpus")
    collections = registry.get_source_database_collections("legal_corpus", "documents")
    entities    = registry.get_source_database_collection_entities("legal_corpus", "documents", "legal_corpus")

    print(f"  Databases   : {[r['name'] for r in databases]}")
    print(f"  Collections : {[r['name'] for r in collections]}")
    print(f"  Entities    : {[r['name'] for r in entities]}")

    if entities:
        attrs = registry.get_source_database_collection_entity_attributes(
            "legal_corpus", "documents", "legal_corpus", entities[0]["name"]
        )
        print(f"  Attributes on '{entities[0]['name']}' ({len(attrs)} total):")
        for a in attrs[:5]:
            print(f"    {a['name']:<20} {a.get('description', '')[:60]}")
        if len(attrs) > 5:
            print(f"    … and {len(attrs) - 5} more")

    print()

    # -----------------------------------------------------------------------
    # 5. search_records — search registry metadata
    # -----------------------------------------------------------------------
    print(DIV2)
    print("Step 5 — search_records (find sources by metadata)")
    print(DIV2)

    for query in ["document chunks", "invoice billing"]:
        hits = registry.search_records(query, page_size=3)
        print(f"  query={query!r}  →  {len(hits)} result(s)")
        for h in hits[:2]:
            print(f"    [{h.get('type','?')}] {h.get('name','?')}  scope={h.get('scope','')}")
    print()

    # -----------------------------------------------------------------------
    # 6. execute_query via registry — structured DOCUMENT.* ops
    # -----------------------------------------------------------------------
    print(DIV2)
    print("Step 6 — execute_query via registry (DOCUMENT.FIND)")
    print(DIV2)

    contracts = registry.execute_query(
        source="legal_corpus",
        query={"op": "DOCUMENT.FIND", "filter": {"document_type": "contract"}},
    )
    print(f"  DOCUMENT.FIND document_type=contract → {len(contracts)} record(s)")
    for rec in contracts:
        print(f"    {rec.get('document_id')}  type={rec.get('document_type')}  confidence={rec.get('confidence', 0):.2f}")

    print()
    chunks = registry.execute_query(
        source="legal_corpus",
        query={
            "op":         "DOCUMENT.SEARCH",
            "query":      "termination clause",
            "collection": "legal_corpus",
            "k":          2,
        },
    )
    print(f"  DOCUMENT.SEARCH 'termination clause' → {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"    [{c.get('document_type','?')}] {c.get('text','')[:80]}…")

    print()

    # -----------------------------------------------------------------------
    # 7. execute_query_nl via registry — natural language dispatch
    # -----------------------------------------------------------------------
    print(DIV2)
    print("Step 7 — execute_query_nl (NL dispatch — BM25, no embedding cost)")
    print(DIV2)
    print("  Note: NLQueryParser requires OPENAI_API_KEY.  Skipping NL demo.")
    print("  To enable, add nl_parser=NLQueryParser() to DocumentDataSource")
    print("  and set OPENAI_API_KEY.  The registry call is then:")
    print()
    print("    result = registry.execute_query_nl(")
    print('        source="legal_corpus",')
    print('        text="find the termination clause in contracts",')
    print("    )")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(DIV)
    print("Result")
    print(DIV)
    print()
    print("  DocumentDataSource participates in DataRegistry like any other source:")
    print("  - register_source  → source is addressable by name")
    print("  - sync_source      → registry learns ChunkRecord schema automatically")
    print("  - search_records   → registry metadata is searchable")
    print("  - execute_query    → DOCUMENT.* ops run via the registry interface")
    print("  - execute_query_nl → NL dispatch when NLQueryParser is wired up")
    print()
    print("  Agents built on DataRegistry can retrieve document evidence and")
    print("  cross-reference it with SQL, graph, or JSON sources — no per-")
    print("  integration glue code required.")
    print()


if __name__ == "__main__":
    main()
