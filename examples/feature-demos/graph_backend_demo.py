"""
Graph backend — entity-centric retrieval and structural navigation
==================================================================
Shows how documents are indexed into a rich knowledge graph during ingest
and queried by entity traversal, sequential navigation, and general
graph ops — without embeddings.

Story:
  A compliance team has three documents — an employment contract, a vendor
  agreement, and an incident report — all mentioning overlapping entities
  (people, companies, dates). They want to ask:
    "What documents mention Acme Corp?"
    "Which chunks reference Sarah Chen?"
    "Show the chunk before and after this finding."
    "Which documents does 'Acme Corp' appear in directly?"

  Vector search can't reliably answer exact-name questions because short
  entity strings don't dominate embedding space. Graph traversal answers
  them without scoring every chunk — the entity node ID is constructed
  deterministically, so no corpus scan is needed.

Full graph schema written during ingest
---------------------------------------
(Document {document_id, file_name, document_type,
           mime_type, ingested_at, chunk_count})
    -[:HAS_SECTION]->
(Section {section_id, title, document_id, position})
    -[:HAS_CHUNK]->
(Chunk {chunk_id, document_id, text_preview, chunk_index, page?, section_title?})
    -[:MENTIONS]->
(Entity {name, entity_type})

Additional edges:
  Document  -[:HAS_CHUNK]->       Chunk    (direct, regardless of section)
  Chunk     -[:NEXT]->            Chunk    (sequential forward)
  Chunk     -[:PREV]->            Chunk    (sequential backward)
  Entity    -[:CO_OCCURS_WITH]->  Entity   (pairs co-occurring in same chunk)
  Entity    -[:APPEARS_IN]->      Document (direct doc-level presence)

What this demo covers:
  * DocumentPipeline(graph_source=MemoryGraphSource(...))
  * Automatic node and edge creation during ingest
  * Section nodes — Document → Section → Chunk hierarchy
  * Sequential navigation — Chunk -[:NEXT]-> Chunk / -[:PREV]-> Chunk
  * Direct entity presence — Entity -[:APPEARS_IN]-> Document
  * Richer Document node: mime_type, ingested_at, chunk_count
  * pipeline.graph_neighbors() — entity-centric chunk retrieval
  * pipeline.graph_query() — general graph traversal
  * DOCUMENT.GRAPH_NEIGHBORS and DOCUMENT.GRAPH_QUERY ops
  * CO_OCCURS_WITH edges between entities in the same chunk
  * Delete cascade: Document + Section + Chunk nodes removed; Entity retained
  * Neo4j production swap (no code change)

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


def _print_chunk_results(results: list[dict]) -> None:
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


def _print_graph_results(results: list[dict]) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        props = r.get("properties", {})
        preview = str(props.get("text_preview", "") or props.get("title", "") or props.get("name", ""))
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        print(f"  [{r['label']}]  {r['node_id']}")
        if preview:
            print(f"           {preview!r}")
        print()


def run() -> None:
    _banner("Graph backend — entity-centric retrieval and structural navigation")

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

    doc_nodes     = graph.match_nodes("documents", "Document")
    chunk_nodes   = graph.match_nodes("documents", "Chunk")
    section_nodes = graph.match_nodes("documents", "Section")
    entity_nodes  = graph.match_nodes("documents", "Entity")

    print()
    print("  Graph state after ingest:")
    print(f"    Document nodes : {len(doc_nodes)}")
    print(f"    Section nodes  : {len(section_nodes)}")
    print(f"    Chunk nodes    : {len(chunk_nodes)}")
    print(f"    Entity nodes   : {len(entity_nodes)}")
    print()
    print("  Entity nodes (all shared across documents):")
    for e in sorted(entity_nodes, key=lambda n: n["properties"]["entity_type"]):
        props = e["properties"]
        print(f"    [{props['entity_type']:<12}]  {props['name']}")

    # ── Step 3: Richer Document node ──────────────────────────────────────
    _step(3, "Richer Document node — mime_type, ingested_at, chunk_count")
    print()
    print("  Document nodes now carry provenance metadata set at ingest time:")
    print()
    for node in doc_nodes:
        p = node["properties"]
        print(f"  {p['file_name']}")
        print(f"    document_id  : {p['document_id']}")
        print(f"    document_type: {p['document_type']}")
        print(f"    mime_type    : {p['mime_type']}")
        print(f"    ingested_at  : {p['ingested_at'] or '(not set in ingest_text)'}")
        print(f"    chunk_count  : {p['chunk_count']}")
        print()

    # ── Step 4: Section hierarchy ─────────────────────────────────────────
    _step(4, "Section nodes — Document -[:HAS_SECTION]-> Section -[:HAS_CHUNK]-> Chunk")
    print()
    print("  Each distinct section_title per document becomes a Section node.")
    print("  Section nodes sit between the Document and its Chunks in the graph,")
    print("  preserving document structure for structural navigation.")
    print()
    if section_nodes:
        for node in section_nodes:
            p = node["properties"]
            print(f"  Section: {p['title']!r}  (position={p['position']}, doc={p['document_id']})")
            # Find chunks under this section
            chunk_refs = graph.get_neighbors(
                "documents", node["id"], edge_type="HAS_CHUNK", direction="out"
            )
            for item in chunk_refs:
                chunk_props = item.get("node", {}).get("properties", {})
                preview = chunk_props.get("text_preview", "").replace("\n", " ").strip()[:50]
                print(f"    └─ chunk {chunk_props.get('chunk_index')}  {preview!r}")
    else:
        print("  (no section headers detected in this corpus — SectionChunker fallback)")

    # ── Step 5: Entity traversal — Acme Corp ──────────────────────────────
    _step(5, "graph_neighbors('Acme Corp') — cross-document entity traversal")
    print()
    print("  pipeline.graph_neighbors('Acme Corp')")
    print()
    print("  Under the hood: entity:PROPER_NOUN:acme corp → MENTIONS edges (in) → Chunks")
    print("  Deterministic node ID lookup — no KNN, no embeddings, no corpus scan.")
    print()

    results = pipeline.graph_neighbors("Acme Corp", k=10)
    _print_chunk_results(results)

    doc_ids = {r["document_id"] for r in results}
    print(f"  → Found in {len(doc_ids)} document(s) across {len(results)} chunk(s)")
    print("    Vector search would need to score all chunks by embedding distance.")
    print("    Graph traversal answers this without scoring any chunks — no embeddings needed.")

    # ── Step 6: Sequential navigation — NEXT / PREV ───────────────────────
    _step(6, "Sequential navigation — Chunk -[:NEXT]-> Chunk / Chunk -[:PREV]-> Chunk")
    print()
    print("  Every consecutive pair of chunks in a document is linked with NEXT and PREV.")
    print("  Walk the chain to expand context around a hit — without storing full text in")
    print("  each node (text_preview is capped at 200 chars for node weight).")
    print()

    next_edges = graph.match_edges("documents", edge_type="NEXT")
    print(f"  NEXT edges in corpus: {len(next_edges)}")
    print()

    # Walk the NEXT chain for the first document
    first_doc = doc_nodes[0]["properties"]["document_id"]
    first_fname = doc_nodes[0]["properties"]["file_name"]
    doc_chunks = graph.get_neighbors(
        "documents", f"doc:{first_doc}", edge_type="HAS_CHUNK", direction="out"
    )
    first_chunk_id = None
    for item in doc_chunks:
        node = item.get("node", {})
        if node.get("properties", {}).get("chunk_index") == 0:
            first_chunk_id = node.get("id")
            break

    if first_chunk_id:
        print(f"  Walking NEXT chain from first chunk of '{first_fname}':")
        current = first_chunk_id
        steps = 0
        while current and steps < 6:
            node = graph.get_node("documents", current)
            props = node.get("properties", {}) if node else {}
            preview = props.get("text_preview", "").replace("\n", " ").strip()[:55]
            print(f"    [{steps}] {current}  {preview!r}")
            nxt = graph.get_neighbors("documents", current, edge_type="NEXT", direction="out")
            current = nxt[0]["node"]["id"] if nxt else None
            steps += 1

    print()
    print("  Use DOCUMENT.GRAPH_QUERY to walk this chain programmatically:")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_QUERY',")
    print("        'node_id': 'chunk:doc_abc_chunk_000003',")
    print("        'edge_type': 'NEXT',")
    print("        'direction': 'out',")
    print("    })")

    # ── Step 7: APPEARS_IN — direct entity → document ─────────────────────
    _step(7, "Entity -[:APPEARS_IN]-> Document — direct document membership")
    print()
    print("  Every entity gets a direct APPEARS_IN edge to each Document it appears in.")
    print("  This lets you answer 'which documents mention Acme Corp?' in one hop,")
    print("  rather than a two-hop traversal (entity → chunk → document).")
    print()

    appears_in_edges = graph.match_edges("documents", edge_type="APPEARS_IN")
    print(f"  APPEARS_IN edges in corpus: {len(appears_in_edges)}")
    print()

    # Use graph_query to show APPEARS_IN from a specific entity
    acme_node_id = "entity:PROPER_NOUN:acme corp"
    if graph.get_node("documents", acme_node_id):
        doc_refs = pipeline.graph_query(
            acme_node_id, edge_type="APPEARS_IN", direction="out"
        )
        print(f"  Documents that 'Acme Corp' APPEARS_IN ({len(doc_refs)} document(s)):")
        for r in doc_refs:
            file_name = r["properties"].get("file_name", "")
            print(f"    → [{r['label']}]  {r['node_id']}  ({file_name})")
    else:
        print("  ('Acme Corp' not found in graph — check BasicEnricher PROPER_NOUN regex)")

    print()
    print("  Compare: the old two-hop path (entity → MENTIONS ← chunk → HAS_CHUNK ← doc)")
    print("  is still valid but APPEARS_IN makes document membership a single hop.")

    # ── Step 8: CO_OCCURS_WITH edges ──────────────────────────────────────
    _step(8, "CO_OCCURS_WITH — entities co-occurring in the same chunk")
    print()
    print("  Entities that appear in the same chunk get a CO_OCCURS_WITH edge.")
    print("  Enables: 'what else was mentioned with Acme Corp in this chunk?'")
    print()

    co_edges = graph.match_edges("documents", edge_type="CO_OCCURS_WITH")
    entity_pairs: set = set()
    for edge in co_edges:
        src = edge["source_id"].split(":", 2)[-1]
        tgt = edge["target_id"].split(":", 2)[-1]
        entity_pairs.add(tuple(sorted([src, tgt])))

    print(f"  {len(co_edges)} CO_OCCURS_WITH edge(s) — {len(entity_pairs)} unique pair(s):")
    for src, tgt in sorted(entity_pairs)[:8]:
        print(f"    {src!r}  <-->  {tgt!r}")

    # ── Step 9: DOCUMENT.GRAPH_QUERY — general traversal ──────────────────
    _step(9, "DOCUMENT.GRAPH_QUERY — general graph traversal via op interface")
    print()
    print("  DOCUMENT.GRAPH_QUERY is a general-purpose traversal op. Start from any")
    print("  node ID, follow any edge type, in any direction, with an optional limit.")
    print()
    print("  Examples:")
    print()
    print("    # All chunks in a document")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_QUERY',")
    print("        'node_id': 'doc:doc_abc123',")
    print("        'edge_type': 'HAS_CHUNK',")
    print("        'direction': 'out',")
    print("    })")
    print()
    print("    # Documents that mention a given entity (APPEARS_IN)")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_QUERY',")
    print("        'node_id': 'entity:PROPER_NOUN:acme corp',")
    print("        'edge_type': 'APPEARS_IN',")
    print("        'direction': 'out',")
    print("    })")
    print()
    print("    # Next chunk in reading order")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_QUERY',")
    print("        'node_id': 'chunk:doc_abc123_chunk_000003',")
    print("        'edge_type': 'NEXT',")
    print("        'direction': 'out',")
    print("        'k': 1,")
    print("    })")
    print()
    print("    # All sections in a document")
    print("    source.execute_query({")
    print("        'op': 'DOCUMENT.GRAPH_QUERY',")
    print("        'node_id': 'doc:doc_abc123',")
    print("        'edge_type': 'HAS_SECTION',")
    print("        'direction': 'out',")
    print("    })")
    print()

    # Live demo: walk HAS_CHUNK from first document
    first_doc_node_id = f"doc:{doc_nodes[0]['properties']['document_id']}"
    results_gq = pipeline.graph_query(
        first_doc_node_id, edge_type="HAS_CHUNK", direction="out", k=3
    )
    print(f"  Live: graph_query('{first_doc_node_id}', edge_type='HAS_CHUNK', k=3)")
    print(f"  → {len(results_gq)} result(s) (capped at k=3):")
    _print_graph_results(results_gq)

    # ── Step 10: Delete cascade ────────────────────────────────────────────
    _step(10, "Delete cascade — Document + Section + Chunk removed; Entity nodes retained")
    print()

    first_doc_id   = doc_nodes[0]["properties"]["document_id"]
    first_doc_fname = doc_nodes[0]["properties"]["file_name"]

    print(f"  Deleting: {first_doc_fname}  ({first_doc_id})")
    pipeline._graph_indexer.delete_document(first_doc_id)

    remaining_docs     = graph.match_nodes("documents", "Document")
    remaining_sections = graph.match_nodes("documents", "Section")
    remaining_chunks   = graph.match_nodes("documents", "Chunk")
    remaining_ents     = graph.match_nodes("documents", "Entity")

    print(f"  After delete:")
    print(f"    Document nodes : {len(remaining_docs)}  (was {len(doc_nodes)})")
    print(f"    Section nodes  : {len(remaining_sections)}  (was {len(section_nodes)})")
    print(f"    Chunk nodes    : {len(remaining_chunks)}  (was {len(chunk_nodes)})")
    print(f"    Entity nodes   : {len(remaining_ents)}  (unchanged — shared across docs)")

    # ── Step 11: Production path — Neo4j ──────────────────────────────────
    _step(11, "Production path — swap MemoryGraphSource for NEO4JSource")
    print()
    print("  Replace MemoryGraphSource with NEO4JSource for persistent storage.")
    print("  No other code changes — the same GraphIndexer, same node IDs, same edges.")
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
    print("  Full graph schema written during ingest:")
    print()
    print("    (Document {document_id, file_name, document_type,")
    print("               mime_type, ingested_at, chunk_count})")
    print("        -[:HAS_SECTION]->")
    print("    (Section {section_id, title, document_id, position})")
    print("        -[:HAS_CHUNK]->")
    print("    (Chunk {chunk_id, document_id, text_preview, chunk_index})")
    print("        -[:MENTIONS]->")
    print("    (Entity {name, entity_type})")
    print()
    print("    Additional edges:")
    print("      Document  -[:HAS_CHUNK]->       Chunk    (direct, regardless of section)")
    print("      Chunk     -[:NEXT]->            Chunk    (sequential forward)")
    print("      Chunk     -[:PREV]->            Chunk    (sequential backward)")
    print("      Entity    -[:CO_OCCURS_WITH]->  Entity   (co-occurrence in same chunk)")
    print("      Entity    -[:APPEARS_IN]->      Document (direct doc membership)")
    print()
    print("  What you get that vector search alone cannot provide:")
    print("    * Exact entity lookup — deterministic node ID, no corpus scan, no embedding")
    print("    * Cross-document links — one entity node shared across all documents")
    print("    * Section hierarchy — Document → Section → Chunk navigable structure")
    print("    * Sequential navigation — walk reading order via NEXT/PREV")
    print("    * Direct document membership — APPEARS_IN skips the chunk hop")
    print("    * Co-occurrence graph — entities that appear together in the same chunk")
    print()
    print("  All existing pipeline behaviour is unchanged — graph_source=None")
    print("  (the default) disables the graph layer with zero overhead.")
    print()
    print("  Backends supported (all from blue-data, no extra code needed):")
    print("    MemoryGraphSource  — dev/test, in-process, no infra")
    print("    FileGraphSource    — local persistence")
    print("    NEO4JSource        — production, Bolt protocol")
    print()


if __name__ == "__main__":
    run()
