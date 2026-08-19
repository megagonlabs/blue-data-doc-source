"""
Graph backend — entity-centric retrieval
=========================================
Shows how to index documents into a graph (Document→Chunk→Entity nodes)
and query by entity traversal instead of vector similarity.

Story:
  A compliance team has three documents — an employment contract, a vendor
  agreement, and an incident report — all mentioning overlapping entities
  (people, companies, dates). They want to ask:
    "What documents mention Acme Corp?"
    "Which chunks reference Sarah Chen?"
    "Show everything that co-occurs with the date January 15, 2026."

  Vector search can't reliably answer these because exact entity names are
  often too short or too common to dominate embedding space. Graph traversal
  answers them in O(1) regardless of corpus size.

What this demo covers:
  * DocumentPipeline(graph_source=MemoryGraphSource(...))
  * Automatic Document/Chunk/Entity node creation during ingest
  * BasicEnricher → PROPER_NOUN + DATE entities → graph nodes
  * pipeline.graph_neighbors() — entity traversal
  * Cross-document entity linking (same entity, multiple documents)
  * CO_OCCURS_WITH edges between entities in the same chunk
  * Delete cascade: Document+Chunk nodes removed, Entity nodes retained
  * DOCUMENT.GRAPH_NEIGHBORS op via DocumentDataSource

Run:
    python3 examples/feature-demos/graph_backend_demo.py

    # For Neo4j (docker required):
    docker run -p 7687:7687 -e NEO4J_AUTH=none neo4j:community
    NEO4J_URI=bolt://localhost:7687 python3 examples/feature-demos/graph_backend_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import BasicEnricher, DocumentPipeline

DIV  = "=" * 72
DIV2 = "-" * 72

# ---------------------------------------------------------------------------
# Inline corpus — three documents with overlapping entities
# ---------------------------------------------------------------------------

EMPLOYMENT_TEXT = """
EMPLOYMENT AGREEMENT

This Employment Agreement is entered into between Acme Corp and
Sarah Chen (the "Employee") effective January 15, 2026.

Position: Senior Software Engineer
Base Salary: $145,000 per year
Notice Period: 60 days
Jurisdiction: State of California
"""

VENDOR_TEXT = """
VENDOR SERVICES AGREEMENT

This agreement is between Acme Corp (the "Client") and
Apex Data Systems LLC (the "Vendor") dated January 15, 2026.

Services: Data pipeline infrastructure and maintenance.
Contract Value: $240,000 annually
SLA Uptime: 99.5%
Auto-Renewal: Yes, unless cancelled 30 days prior.
"""

INCIDENT_TEXT = """
INCIDENT REPORT  INC-2026-0088

Date: August 1, 2026
Reported By: Sarah Chen, Lab Manager
Assigned To: James Ortega, Facilities

Description: Walk-in freezer failure in Building A. Temperature rose from
-20C to -8C over 4 hours. Samples transferred to backup unit.

Root Cause: Compressor failure — refrigerant leak.
Status: Resolved. Acme Corp notified as equipment vendor.
"""


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


def _print_results(results: list[dict]) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        section = f"  [{r['section_title']}]" if r.get("section_title") else ""
        preview = r["text_preview"].replace("\n", " ").strip()
        if len(preview) > 70:
            preview = preview[:67] + "..."
        print(f"  chunk: {r['chunk_id']}")
        print(f"  doc:   {r['document_id']}{section}")
        print(f"  text:  {preview!r}")
        print()


def run() -> None:
    _banner("Graph backend — entity-centric retrieval")

    print()
    print("  Three documents with overlapping entities:")
    print("    employment_chen.txt  — Acme Corp, Sarah Chen, January 15 2026")
    print("    vendor_apex.txt      — Acme Corp, Apex Data Systems, January 15 2026")
    print("    incident_0088.txt    — Sarah Chen, James Ortega, Acme Corp")
    print()
    print("  Backend: MemoryGraphSource (in-process, no infra required)")
    print("  Entities: BasicEnricher extracts PROPER_NOUN and DATE automatically")

    # ── Step 1: Build the pipeline ────────────────────────────────────────
    _step(1, "Build pipeline with graph_source=MemoryGraphSource()")
    print()
    print("  graph = MemoryGraphSource('demo')")
    print("  pipeline = DocumentPipeline(")
    print("      enricher=BasicEnricher(),   # extracts entities from each chunk")
    print("      graph_source=graph,          # enables graph indexing")
    print("      graph_database='documents',")
    print("  )")
    print()

    graph = MemoryGraphSource("demo")
    pipeline = DocumentPipeline(
        chunk_size=80, overlap=10,
        enricher=BasicEnricher(),
        graph_source=graph,
        graph_database="documents",
    )

    # ── Step 2: Ingest the three documents ────────────────────────────────
    _step(2, "Ingest three documents — graph nodes created automatically")
    print()

    docs = [
        ("employment_chen.txt", EMPLOYMENT_TEXT),
        ("vendor_apex.txt",     VENDOR_TEXT),
        ("incident_0088.txt",   INCIDENT_TEXT),
    ]
    for fname, text in docs:
        chunks = pipeline.ingest_text(text, file_name=fname)
        print(f"  {fname}  →  {len(chunks)} chunk(s)")

    # Show what landed in the graph
    doc_nodes    = graph.match_nodes("documents", "Document")
    chunk_nodes  = graph.match_nodes("documents", "Chunk")
    entity_nodes = graph.match_nodes("documents", "Entity")

    print()
    print(f"  Graph state after ingest:")
    print(f"    Document nodes : {len(doc_nodes)}")
    print(f"    Chunk nodes    : {len(chunk_nodes)}")
    print(f"    Entity nodes   : {len(entity_nodes)}")
    print()
    print("  Entity nodes (sample — all shared across documents):")
    for e in sorted(entity_nodes, key=lambda n: n["properties"]["entity_type"]):
        props = e["properties"]
        print(f"    [{props['entity_type']:<12}]  {props['name']}")

    # ── Step 3: Entity traversal — Acme Corp ────────────────────
    _step(3, "graph_neighbors('Acme Corp') — cross-document traversal")
    print()
    print("  pipeline.graph_neighbors('Acme Corp')")
    print()

    results = pipeline.graph_neighbors("Acme Corp", k=10)
    _print_results(results)

    doc_ids = {r["document_id"] for r in results}
    print(f"  → Found in {len(doc_ids)} document(s) across {len(results)} chunk(s)")
    print("    Vector search would need to score all chunks by embedding distance.")
    print("    Graph traversal answers this in O(1) — no embeddings needed.")

    # ── Step 4: Entity traversal — Sarah Chen ────────────────────────────
    _step(4, "graph_neighbors('Sarah Chen') — person entity across two documents")
    print()
    print("  pipeline.graph_neighbors('Sarah Chen')")
    print()

    results_sc = pipeline.graph_neighbors("Sarah Chen", k=10)
    _print_results(results_sc)

    # ── Step 5: Date entity lookup ────────────────────────────────────────
    _step(5, "graph_neighbors('January 15, 2026', entity_type='DATE')")
    print()
    print("  BasicEnricher tagged this date in both the employment and vendor docs.")
    print()

    results_date = pipeline.graph_neighbors(
        "January 15, 2026", entity_type="DATE", k=10
    )
    _print_results(results_date)

    # ── Step 6: CO_OCCURS_WITH edges ─────────────────────────────────────
    _step(6, "CO_OCCURS_WITH edges — entities co-occurring in the same chunk")
    print()
    print("  Entities that appear in the same chunk get a CO_OCCURS_WITH edge.")
    print("  This enables questions like 'what else was mentioned with Sarah Chen?'")
    print()

    co_edges = graph.match_edges("documents", edge_type="CO_OCCURS_WITH")
    entity_pairs = set()
    for edge in co_edges:
        src = edge["source_id"].split(":", 2)[-1]   # strip "entity:TYPE:" prefix
        tgt = edge["target_id"].split(":", 2)[-1]
        pair = tuple(sorted([src, tgt]))
        entity_pairs.add(pair)

    print(f"  {len(co_edges)} CO_OCCURS_WITH edge(s) — {len(entity_pairs)} unique pair(s):")
    for src, tgt in sorted(entity_pairs)[:8]:
        print(f"    {src!r}  <-->  {tgt!r}")

    # ── Step 7: Delete cascade ────────────────────────────────────────────
    _step(7, "Delete cascade — Document+Chunk removed, Entity nodes retained")
    print()

    # Pick the first ingested doc
    first_doc_id = doc_nodes[0]["properties"]["document_id"]
    first_fname  = doc_nodes[0]["properties"]["file_name"]

    print(f"  Deleting: {first_fname}  ({first_doc_id})")
    pipeline._graph_indexer.delete_document(first_doc_id)

    remaining_docs   = graph.match_nodes("documents", "Document")
    remaining_chunks = graph.match_nodes("documents", "Chunk")
    remaining_ents   = graph.match_nodes("documents", "Entity")

    print(f"  After delete:")
    print(f"    Document nodes : {len(remaining_docs)}  (was {len(doc_nodes)})")
    print(f"    Chunk nodes    : {len(remaining_chunks)}  (was {len(chunk_nodes)})")
    print(f"    Entity nodes   : {len(remaining_ents)}  (unchanged — shared across docs)")

    # ── Step 8: DOCUMENT.GRAPH_NEIGHBORS via DocumentDataSource ──────────
    _step(8, "DOCUMENT.GRAPH_NEIGHBORS op via DocumentDataSource")
    print()
    print("  The same traversal is available as a structured op:")
    print()
    print("    from blue_data_doc_source import DocumentDataSource")
    print("    source = DocumentDataSource(")
    print("        enricher=BasicEnricher(),")
    print("        graph_source=MemoryGraphSource('prod'),")
    print("    )")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_NEIGHBORS',")
    print("        'entity': 'Acme Corp',")
    print("        'k': 5,")
    print("    })")
    print()
    print("  Optional: 'entity_type' restricts to PROPER_NOUN, DATE, MONEY, etc.")

    # ── Step 9: Neo4j production path ────────────────────────────────────
    _step(9, "Production path — swap MemoryGraphSource for NEO4JSource")
    print()
    print("  Replace MemoryGraphSource with NEO4JSource for persistent storage:")
    print()
    print("    from blue_data.source.graph.neo4j.neo4j_graph_source import NEO4JSource")
    print()
    print("    neo4j = NEO4JSource('neo4j', properties={")
    print("        'connection': {")
    print("            'host': 'localhost',")
    print("            'port': 7687,")
    print("            'user': 'neo4j',")
    print("            'password': 'password',")
    print("        }")
    print("    })")
    print("    pipeline = DocumentPipeline(")
    print("        enricher=BasicEnricher(),")
    print("        graph_source=neo4j,")
    print("        graph_database='documents',")
    print("    )")
    print()
    print("  Quick start (Docker):")
    print("    docker run -p 7687:7687 -e NEO4J_AUTH=none neo4j:community")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  Graph schema written during ingest:")
    print("    (Document) -[:HAS_CHUNK]->  (Chunk)")
    print("    (Chunk)    -[:MENTIONS]->   (Entity)")
    print("    (Entity)   -[:CO_OCCURS_WITH]-> (Entity)")
    print()
    print("  What you get that vector search can't provide:")
    print("    * Exact entity lookup — O(1), no embedding required")
    print("    * Cross-document links — same entity, multiple documents")
    print("    * Co-occurrence graph — entities that appear together")
    print("    * Provenance path — why a chunk was retrieved (entity trail)")
    print()
    print("  All existing pipeline behaviour is unchanged — graph_source=None")
    print("  (the default) disables the graph layer with zero overhead.")
    print()
    print("  Backends supported (all from blue-data, no extra code):")
    print("    MemoryGraphSource  — dev/test, in-process")
    print("    FileGraphSource    — local persistence")
    print("    NEO4JSource        — production")
    print()


if __name__ == "__main__":
    run()
