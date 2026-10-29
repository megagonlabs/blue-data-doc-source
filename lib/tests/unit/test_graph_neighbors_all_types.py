"""Regression tests for issue #88 (DOCSRC-20).

graph_neighbors without an entity_type only probed the BasicEnricher type set, so
entities typed by LLMEnricher (ORG, PERSON, LOCATION, …) were never found and the
lookup returned [] silently. It now matches by name across all entity types.
"""
from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data_doc_source import GraphIndexer


def _indexer() -> GraphIndexer:
    g = MemoryGraphSource("t")
    g.create_database("docs")
    return GraphIndexer(g, database="docs")


def _records(entity_type: str):
    return [{
        "chunk_id": "doc1_chunk_000000",
        "document_id": "doc1",
        "text": "Agreement between Acme Corp and Beta LLC.",
        "entities": [{"text": "Acme Corp", "type": entity_type}],
    }]


def test_without_type_finds_llm_entity_type():
    idx = _indexer()
    idx.index_document("doc1", "c.txt", None, _records("ORG"))

    no_type = idx.graph_neighbors("Acme Corp")
    typed = idx.graph_neighbors("Acme Corp", entity_type="ORG")

    assert len(no_type) == 1
    assert no_type == typed
    assert no_type[0]["entity_type"] == "ORG"


def test_without_type_is_case_insensitive():
    idx = _indexer()
    idx.index_document("doc1", "c.txt", None, _records("PERSON"))
    assert len(idx.graph_neighbors("acme corp")) == 1


def test_wrong_explicit_type_still_empty():
    idx = _indexer()
    idx.index_document("doc1", "c.txt", None, _records("ORG"))
    assert idx.graph_neighbors("Acme Corp", entity_type="PROPER_NOUN") == []


def test_unknown_name_returns_empty():
    idx = _indexer()
    idx.index_document("doc1", "c.txt", None, _records("ORG"))
    assert idx.graph_neighbors("Nonexistent Inc") == []
