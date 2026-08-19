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
