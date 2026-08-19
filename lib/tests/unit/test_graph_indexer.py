"""Unit tests for GraphIndexer using MemoryGraphSource (no external dependencies)."""
import pytest

from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import BasicEnricher, DocumentPipeline, GraphIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph() -> MemoryGraphSource:
    g = MemoryGraphSource("test")
    g.create_database("docs")
    return g


def _make_indexer(graph=None) -> tuple[GraphIndexer, MemoryGraphSource]:
    g = graph or _make_graph()
    return GraphIndexer(g, database="docs"), g


def _simple_records(document_id: str = "doc_abc") -> list[dict]:
    return [
        {
            "chunk_id": f"{document_id}_chunk_000000",
            "document_id": document_id,
            "text": "NovaTech Solutions signed the agreement on January 15, 2026.",
            "entities": [
                {"text": "NovaTech Solutions", "type": "PROPER_NOUN"},
                {"text": "January 15, 2026", "type": "DATE"},
            ],
            "section_title": "Introduction",
        },
        {
            "chunk_id": f"{document_id}_chunk_000001",
            "document_id": document_id,
            "text": "Sarah Chen will lead the project.",
            "entities": [
                {"text": "Sarah Chen", "type": "PROPER_NOUN"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# GraphIndexer unit tests
# ---------------------------------------------------------------------------

class TestGraphIndexerNodes:
    def test_document_node_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "contract.txt", "contract", _simple_records())

        node = graph.get_node("docs", "doc:doc_abc")
        assert node is not None
        assert node["label"] == "Document"
        assert node["properties"]["document_id"] == "doc_abc"
        assert node["properties"]["file_name"] == "contract.txt"
        assert node["properties"]["document_type"] == "contract"

    def test_chunk_nodes_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "contract.txt", None, _simple_records())

        chunk0 = graph.get_node("docs", "chunk:doc_abc_chunk_000000")
        assert chunk0 is not None
        assert chunk0["label"] == "Chunk"
        assert chunk0["properties"]["chunk_id"] == "doc_abc_chunk_000000"
        assert chunk0["properties"]["section_title"] == "Introduction"

        chunk1 = graph.get_node("docs", "chunk:doc_abc_chunk_000001")
        assert chunk1 is not None
        assert "section_title" not in chunk1["properties"]

    def test_entity_nodes_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        novatech = graph.get_node("docs", "entity:PROPER_NOUN:novatech solutions")
        assert novatech is not None
        assert novatech["properties"]["name"] == "NovaTech Solutions"
        assert novatech["properties"]["entity_type"] == "PROPER_NOUN"

        date_node = graph.get_node("docs", "entity:DATE:january 15, 2026")
        assert date_node is not None

    def test_document_type_unknown_when_none(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())
        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["document_type"] == "unknown"

    def test_text_preview_truncated_to_200_chars(self):
        indexer, graph = _make_indexer()
        long_text = "word " * 100  # 500 chars
        records = [{
            "chunk_id": "doc_x_chunk_000000",
            "document_id": "doc_x",
            "text": long_text,
            "entities": [],
        }]
        indexer.index_document("doc_x", "f.txt", None, records)
        chunk = graph.get_node("docs", "chunk:doc_x_chunk_000000")
        assert len(chunk["properties"]["text_preview"]) <= 200


class TestGraphIndexerEdges:
    def test_has_chunk_edges(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        edges = graph.match_edges("docs", edge_type="HAS_CHUNK", source_id="doc:doc_abc")
        target_ids = {e["target_id"] for e in edges}
        assert "chunk:doc_abc_chunk_000000" in target_ids
        assert "chunk:doc_abc_chunk_000001" in target_ids

    def test_mentions_edges(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        mentions = graph.match_edges(
            "docs", edge_type="MENTIONS", source_id="chunk:doc_abc_chunk_000000"
        )
        target_ids = {e["target_id"] for e in mentions}
        assert "entity:PROPER_NOUN:novatech solutions" in target_ids
        assert "entity:DATE:january 15, 2026" in target_ids

    def test_co_occurs_with_edges(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        # chunk 0 has two entities — they should be CO_OCCURS_WITH each other
        co = graph.match_edges("docs", edge_type="CO_OCCURS_WITH")
        assert len(co) >= 1


class TestGraphIndexerIdempotency:
    def test_re_index_same_document_is_noop(self):
        indexer, graph = _make_indexer()
        records = _simple_records()
        indexer.index_document("doc_abc", "f.txt", "contract", records)
        indexer.index_document("doc_abc", "f.txt", "contract", records)

        # Only one Document node
        doc_nodes = graph.match_nodes("docs", "Document")
        assert len(doc_nodes) == 1

    def test_entity_node_shared_across_documents(self):
        indexer, graph = _make_indexer()

        records_a = [{
            "chunk_id": "doc_a_chunk_000000",
            "document_id": "doc_a",
            "text": "NovaTech Solutions is the vendor.",
            "entities": [{"text": "NovaTech Solutions", "type": "PROPER_NOUN"}],
        }]
        records_b = [{
            "chunk_id": "doc_b_chunk_000000",
            "document_id": "doc_b",
            "text": "NovaTech Solutions signed the SLA.",
            "entities": [{"text": "NovaTech Solutions", "type": "PROPER_NOUN"}],
        }]
        indexer.index_document("doc_a", "a.txt", None, records_a)
        indexer.index_document("doc_b", "b.txt", None, records_b)

        entity_nodes = graph.match_nodes("docs", "Entity", {"name": "NovaTech Solutions"})
        assert len(entity_nodes) == 1  # shared, not duplicated


class TestGraphIndexerDelete:
    def test_delete_removes_document_and_chunks(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())
        indexer.delete_document("doc_abc")

        assert graph.get_node("docs", "doc:doc_abc") is None
        assert graph.get_node("docs", "chunk:doc_abc_chunk_000000") is None
        assert graph.get_node("docs", "chunk:doc_abc_chunk_000001") is None

    def test_delete_retains_entity_nodes(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())
        indexer.delete_document("doc_abc")

        # Entity nodes stay — other documents may reference them
        novatech = graph.get_node("docs", "entity:PROPER_NOUN:novatech solutions")
        assert novatech is not None

    def test_delete_nonexistent_document_is_safe(self):
        indexer, _ = _make_indexer()
        indexer.delete_document("doc_does_not_exist")  # should not raise


class TestGraphNeighbors:
    def test_find_chunks_by_proper_noun(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.graph_neighbors("NovaTech Solutions")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc_abc_chunk_000000"
        assert results[0]["document_id"] == "doc_abc"
        assert results[0]["entity"] == "NovaTech Solutions"
        assert results[0]["entity_type"] == "PROPER_NOUN"

    def test_find_chunks_by_date(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.graph_neighbors("January 15, 2026", entity_type="DATE")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc_abc_chunk_000000"

    def test_case_insensitive_lookup(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        # Node was stored with "NovaTech Solutions" — lookup with any casing
        results_lower = indexer.graph_neighbors("novatech solutions")
        results_upper = indexer.graph_neighbors("NOVATECH SOLUTIONS")
        assert len(results_lower) == 1
        assert len(results_upper) == 1

    def test_unknown_entity_returns_empty(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.graph_neighbors("Acme Corp")
        assert results == []

    def test_entity_across_multiple_documents(self):
        indexer, graph = _make_indexer()
        records_a = [{
            "chunk_id": "doc_a_chunk_000000",
            "document_id": "doc_a",
            "text": "NovaTech Solutions signed contract A.",
            "entities": [{"text": "NovaTech Solutions", "type": "PROPER_NOUN"}],
        }]
        records_b = [{
            "chunk_id": "doc_b_chunk_000000",
            "document_id": "doc_b",
            "text": "NovaTech Solutions is audited.",
            "entities": [{"text": "NovaTech Solutions", "type": "PROPER_NOUN"}],
        }]
        indexer.index_document("doc_a", "a.txt", None, records_a)
        indexer.index_document("doc_b", "b.txt", None, records_b)

        results = indexer.graph_neighbors("NovaTech Solutions")
        assert len(results) == 2
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {"doc_a", "doc_b"}

    def test_k_limits_results(self):
        indexer, graph = _make_indexer()
        records = [
            {
                "chunk_id": f"doc_x_chunk_{i:06d}",
                "document_id": "doc_x",
                "text": f"NovaTech Solutions chunk {i}.",
                "entities": [{"text": "NovaTech Solutions", "type": "PROPER_NOUN"}],
            }
            for i in range(10)
        ]
        indexer.index_document("doc_x", "x.txt", None, records)

        results = indexer.graph_neighbors("NovaTech Solutions", k=3)
        assert len(results) == 3

    def test_section_title_included_when_present(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.graph_neighbors("NovaTech Solutions")
        assert results[0].get("section_title") == "Introduction"

    def test_no_section_title_when_absent(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.graph_neighbors("Sarah Chen")
        assert "section_title" not in results[0]


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelineGraphIntegration:
    def test_ingest_populates_graph(self):
        graph = MemoryGraphSource("test")
        pipeline = DocumentPipeline(
            chunk_size=50, overlap=5,
            enricher=BasicEnricher(),
            graph_source=graph,
            graph_database="docs",
        )
        pipeline.ingest_text(
            "Acme Corp signed the contract. Jane Smith is the signatory.",
            file_name="agreement.txt",
        )

        # Document node must exist
        docs = graph.match_nodes("docs", "Document")
        assert len(docs) == 1

        # Chunk nodes must exist
        chunks = graph.match_nodes("docs", "Chunk")
        assert len(chunks) >= 1

        # At least one entity found by BasicEnricher PROPER_NOUN
        entities = graph.match_nodes("docs", "Entity")
        assert len(entities) >= 1

    def test_graph_neighbors_via_pipeline(self):
        graph = MemoryGraphSource("test")
        pipeline = DocumentPipeline(
            chunk_size=50, overlap=5,
            enricher=BasicEnricher(),
            graph_source=graph,
            graph_database="docs",
        )
        pipeline.ingest_text(
            "Acme Corp and Jane Smith co-signed the service agreement.",
            file_name="svc.txt",
        )

        results = pipeline.graph_neighbors("Acme Corp")
        assert len(results) >= 1
        assert results[0]["entity"] == "Acme Corp"

    def test_graph_neighbors_raises_without_graph_source(self):
        pipeline = DocumentPipeline(chunk_size=50, overlap=5)
        with pytest.raises(ValueError, match="graph_source"):
            pipeline.graph_neighbors("Acme Corp")

    def test_delete_cascades_to_graph(self):
        graph = MemoryGraphSource("test")
        pipeline = DocumentPipeline(
            chunk_size=50, overlap=5,
            enricher=BasicEnricher(),
            graph_source=graph,
            graph_database="docs",
        )
        chunks = pipeline.ingest_text(
            "Acme Corp signed the deal.",
            file_name="deal.txt",
        )
        document_id = chunks[0]["document_id"]

        # Confirm Document node exists
        assert graph.get_node("docs", f"doc:{document_id}") is not None

        # pipeline.delete requires vector_source — call GraphIndexer directly
        pipeline._graph_indexer.delete_document(document_id)
        assert graph.get_node("docs", f"doc:{document_id}") is None

    def test_no_graph_source_no_side_effects(self):
        """Existing pipeline with no graph_source must be unaffected."""
        pipeline = DocumentPipeline(chunk_size=50, overlap=5, enricher=BasicEnricher())
        records = pipeline.ingest_text("Hello world. Some text here.", file_name="t.txt")
        assert len(records) >= 1
        assert pipeline._graph_indexer is None


# ---------------------------------------------------------------------------
# Section nodes
# ---------------------------------------------------------------------------

class TestSectionNodes:
    def test_section_node_created_for_sectioned_chunk(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        section_node = graph.get_node("docs", "section:doc_abc:introduction")
        assert section_node is not None
        assert section_node["label"] == "Section"
        props = section_node["properties"]
        assert props["title"] == "Introduction"
        assert props["document_id"] == "doc_abc"
        assert props["position"] == 0

    def test_section_id_is_deterministic(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        # Node ID matches the pattern section:{doc_id}:{title.lower()}
        node = graph.get_node("docs", "section:doc_abc:introduction")
        assert node["properties"]["section_id"] == "section:doc_abc:introduction"

    def test_no_section_node_when_section_title_absent(self):
        indexer, graph = _make_indexer()
        records = [{
            "chunk_id": "doc_x_chunk_000000",
            "document_id": "doc_x",
            "text": "Plain text, no heading.",
            "entities": [],
        }]
        indexer.index_document("doc_x", "f.txt", None, records)

        section_nodes = graph.match_nodes("docs", "Section")
        assert len(section_nodes) == 0

    def test_same_section_title_creates_one_section_node(self):
        indexer, graph = _make_indexer()
        records = [
            {
                "chunk_id": "doc_x_chunk_000000",
                "document_id": "doc_x",
                "text": "First chunk of the section.",
                "entities": [],
                "section_title": "Overview",
            },
            {
                "chunk_id": "doc_x_chunk_000001",
                "document_id": "doc_x",
                "text": "Second chunk of the same section.",
                "entities": [],
                "section_title": "Overview",
            },
        ]
        indexer.index_document("doc_x", "f.txt", None, records)

        section_nodes = graph.match_nodes("docs", "Section")
        assert len(section_nodes) == 1

    def test_has_section_edge(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        edges = graph.match_edges("docs", edge_type="HAS_SECTION", source_id="doc:doc_abc")
        assert len(edges) == 1
        assert edges[0]["target_id"] == "section:doc_abc:introduction"

    def test_section_has_chunk_edge(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        edges = graph.match_edges(
            "docs", edge_type="HAS_CHUNK", source_id="section:doc_abc:introduction"
        )
        target_ids = {e["target_id"] for e in edges}
        assert "chunk:doc_abc_chunk_000000" in target_ids

    def test_delete_removes_section_nodes(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())
        indexer.delete_document("doc_abc")

        section_nodes = graph.match_nodes("docs", "Section")
        assert len(section_nodes) == 0


# ---------------------------------------------------------------------------
# NEXT / PREV sequential edges
# ---------------------------------------------------------------------------

class TestNextPrevEdges:
    def test_next_edges_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        next_edges = graph.match_edges("docs", edge_type="NEXT")
        assert len(next_edges) == 1
        assert next_edges[0]["source_id"] == "chunk:doc_abc_chunk_000000"
        assert next_edges[0]["target_id"] == "chunk:doc_abc_chunk_000001"

    def test_prev_edges_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        prev_edges = graph.match_edges("docs", edge_type="PREV")
        assert len(prev_edges) == 1
        assert prev_edges[0]["source_id"] == "chunk:doc_abc_chunk_000001"
        assert prev_edges[0]["target_id"] == "chunk:doc_abc_chunk_000000"

    def test_no_next_prev_for_single_chunk(self):
        indexer, graph = _make_indexer()
        records = [{
            "chunk_id": "doc_x_chunk_000000",
            "document_id": "doc_x",
            "text": "Only one chunk.",
            "entities": [],
        }]
        indexer.index_document("doc_x", "f.txt", None, records)

        assert graph.match_edges("docs", edge_type="NEXT") == []
        assert graph.match_edges("docs", edge_type="PREV") == []

    def test_chain_length_matches_chunk_count(self):
        indexer, graph = _make_indexer()
        n = 5
        records = [
            {
                "chunk_id": f"doc_c_chunk_{i:06d}",
                "document_id": "doc_c",
                "text": f"Chunk {i} content.",
                "entities": [],
            }
            for i in range(n)
        ]
        indexer.index_document("doc_c", "f.txt", None, records)

        next_edges = graph.match_edges("docs", edge_type="NEXT")
        prev_edges = graph.match_edges("docs", edge_type="PREV")
        assert len(next_edges) == n - 1
        assert len(prev_edges) == n - 1

    def test_chunk_index_property_stored(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        chunk0 = graph.get_node("docs", "chunk:doc_abc_chunk_000000")
        chunk1 = graph.get_node("docs", "chunk:doc_abc_chunk_000001")
        assert chunk0["properties"]["chunk_index"] == 0
        assert chunk1["properties"]["chunk_index"] == 1


# ---------------------------------------------------------------------------
# APPEARS_IN edges
# ---------------------------------------------------------------------------

class TestAppearsInEdges:
    def test_appears_in_edge_created(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        # Every entity in the doc should have an APPEARS_IN → doc node
        appears_in = graph.match_edges("docs", edge_type="APPEARS_IN")
        target_ids = {e["target_id"] for e in appears_in}
        source_ids = {e["source_id"] for e in appears_in}

        assert "doc:doc_abc" in target_ids
        assert "entity:PROPER_NOUN:novatech solutions" in source_ids
        assert "entity:DATE:january 15, 2026" in source_ids
        assert "entity:PROPER_NOUN:sarah chen" in source_ids

    def test_appears_in_count_equals_unique_entities(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        appears_in = graph.match_edges("docs", edge_type="APPEARS_IN")
        unique_entities = {e["source_id"] for e in appears_in}
        # Three distinct entities across the two chunks
        assert len(unique_entities) == 3

    def test_shared_entity_appears_in_both_documents(self):
        indexer, graph = _make_indexer()
        records_a = [{
            "chunk_id": "doc_a_chunk_000000",
            "document_id": "doc_a",
            "text": "Acme Corp is client A.",
            "entities": [{"text": "Acme Corp", "type": "PROPER_NOUN"}],
        }]
        records_b = [{
            "chunk_id": "doc_b_chunk_000000",
            "document_id": "doc_b",
            "text": "Acme Corp is also client B.",
            "entities": [{"text": "Acme Corp", "type": "PROPER_NOUN"}],
        }]
        indexer.index_document("doc_a", "a.txt", None, records_a)
        indexer.index_document("doc_b", "b.txt", None, records_b)

        appears_in = graph.match_edges("docs", edge_type="APPEARS_IN")
        targets = {e["target_id"] for e in appears_in}
        assert "doc:doc_a" in targets
        assert "doc:doc_b" in targets


# ---------------------------------------------------------------------------
# Richer Document node properties
# ---------------------------------------------------------------------------

class TestDocumentNodeProperties:
    def test_mime_type_stored(self):
        indexer, graph = _make_indexer()
        records = _simple_records()
        records[0]["mime_type"] = "application/pdf"
        indexer.index_document("doc_abc", "f.pdf", None, records)

        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["mime_type"] == "application/pdf"

    def test_mime_type_defaults_to_text_plain(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["mime_type"] == "text/plain"

    def test_chunk_count_stored(self):
        indexer, graph = _make_indexer()
        records = _simple_records()  # 2 records
        indexer.index_document("doc_abc", "f.txt", None, records)

        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["chunk_count"] == 2

    def test_ingested_at_stored_when_present(self):
        indexer, graph = _make_indexer()
        records = _simple_records()
        records[0]["metadata"] = {"ingested_at": "2026-08-19T10:00:00Z"}
        indexer.index_document("doc_abc", "f.txt", None, records)

        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["ingested_at"] == "2026-08-19T10:00:00Z"

    def test_ingested_at_empty_string_when_absent(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        node = graph.get_node("docs", "doc:doc_abc")
        assert node["properties"]["ingested_at"] == ""


# ---------------------------------------------------------------------------
# get_neighbors / DOCUMENT.GRAPH_QUERY
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_get_neighbors_has_chunk(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.get_neighbors("doc:doc_abc", edge_type="HAS_CHUNK", direction="out")
        node_ids = {r["node_id"] for r in results}
        assert "chunk:doc_abc_chunk_000000" in node_ids
        assert "chunk:doc_abc_chunk_000001" in node_ids

    def test_get_neighbors_result_shape(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.get_neighbors("doc:doc_abc", edge_type="HAS_CHUNK", direction="out")
        for r in results:
            assert "node_id" in r
            assert "label" in r
            assert "properties" in r
            assert "edge_type" in r
            assert "direction" in r

    def test_get_neighbors_k_limits_results(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.get_neighbors("doc:doc_abc", edge_type="HAS_CHUNK", direction="out", k=1)
        assert len(results) == 1

    def test_get_neighbors_next_chain(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.get_neighbors(
            "chunk:doc_abc_chunk_000000", edge_type="NEXT", direction="out"
        )
        assert len(results) == 1
        assert results[0]["node_id"] == "chunk:doc_abc_chunk_000001"

    def test_get_neighbors_appears_in(self):
        indexer, graph = _make_indexer()
        indexer.index_document("doc_abc", "f.txt", None, _simple_records())

        results = indexer.get_neighbors(
            "entity:PROPER_NOUN:novatech solutions",
            edge_type="APPEARS_IN",
            direction="out",
        )
        assert len(results) == 1
        assert results[0]["node_id"] == "doc:doc_abc"
        assert results[0]["label"] == "Document"

    def test_graph_query_via_pipeline(self):
        graph = MemoryGraphSource("test")
        pipeline = DocumentPipeline(
            chunk_size=50, overlap=5,
            enricher=BasicEnricher(),
            graph_source=graph,
            graph_database="docs",
        )
        pipeline.ingest_text(
            "Acme Corp and Jane Smith signed the agreement.",
            file_name="agreement.txt",
        )

        docs = graph.match_nodes("docs", "Document")
        doc_id = docs[0]["properties"]["document_id"]

        results = pipeline.graph_query(f"doc:{doc_id}", edge_type="HAS_CHUNK", direction="out")
        assert len(results) >= 1
        assert all(r["label"] == "Chunk" for r in results)

    def test_graph_query_raises_without_graph_source(self):
        pipeline = DocumentPipeline(chunk_size=50, overlap=5)
        with pytest.raises(ValueError, match="graph_source"):
            pipeline.graph_query("doc:xyz")

    def test_document_graph_query_op(self):
        from blue_data_doc_source import DocumentDataSource

        graph = MemoryGraphSource("test")
        source = DocumentDataSource(
            enricher=BasicEnricher(),
            graph_source=graph,
            graph_database="docs",
        )
        # Use pipeline directly — DocumentDataSource.execute_query only accepts files
        source._pipeline.ingest_text(
            "Acme Corp and Jane Smith co-signed the agreement.",
            file_name="agreement.txt",
        )

        docs = graph.match_nodes("docs", "Document")
        doc_id = docs[0]["properties"]["document_id"]

        results = source.execute_query({
            "op": "DOCUMENT.GRAPH_QUERY",
            "node_id": f"doc:{doc_id}",
            "edge_type": "HAS_CHUNK",
            "direction": "out",
        })
        assert len(results) >= 1
        assert all(r["label"] == "Chunk" for r in results)

    def test_document_graph_query_op_missing_node_id(self):
        from blue_data_doc_source import DocumentDataSource

        graph = MemoryGraphSource("test")
        source = DocumentDataSource(graph_source=graph)
        with pytest.raises(ValueError, match="node_id"):
            source.execute_query({"op": "DOCUMENT.GRAPH_QUERY"})
