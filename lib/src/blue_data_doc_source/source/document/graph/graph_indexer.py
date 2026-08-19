from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from blue_data.source.graph.graph_source import GraphSource

# Entity types BasicEnricher can produce — used for deterministic ID lookup.
_KNOWN_ENTITY_TYPES = ("PROPER_NOUN", "DATE", "MONEY", "EMAIL", "PERCENT", "UNKNOWN")


class GraphIndexer:
    """Writes document/chunk/entity nodes into a GraphSource during ingest.

    Graph schema
    ------------
    (Document {document_id, file_name, document_type})
        -[:HAS_CHUNK]->
    (Chunk {chunk_id, document_id, text_preview, section_title?})
        -[:MENTIONS]->
    (Entity {name, entity_type})
    (Entity) -[:CO_OCCURS_WITH]-> (Entity)   [pairs co-occurring in same chunk]

    Node IDs are deterministic:
        Document → ``doc:{document_id}``
        Chunk    → ``chunk:{chunk_id}``
        Entity   → ``entity:{entity_type}:{name.lower()}``

    This means re-ingesting the exact same file is a no-op for the graph
    (the Document node already exists and the call returns early).
    Entity nodes are shared across documents and are never removed by
    ``delete_document`` — only Document and Chunk nodes are removed.
    """

    def __init__(self, graph_source: "GraphSource", database: str = "documents") -> None:
        self._graph = graph_source
        self._db = database

    def ensure_database(self) -> None:
        """Create the graph database if it does not already exist."""
        try:
            self._graph.create_database(self._db, overwrite=False)
        except Exception:
            pass

    # ── ingest ────────────────────────────────────────────────────────────

    def index_document(
        self,
        document_id: str,
        file_name: str,
        document_type: Optional[str],
        records: List[Dict[str, Any]],
    ) -> None:
        """Index a document and all its chunk records into the graph.

        Idempotent: if the Document node already exists this call is a no-op.
        To refresh an existing document call ``delete_document()`` first.

        Args:
            document_id: Content-hash document ID (e.g. ``doc_abc123``).
            file_name: Source file name stored on the Document node.
            document_type: Classifier output or ``None`` / ``"unknown"``.
            records: List of ``ChunkRecord.to_dict()`` dicts produced by the
                pipeline. Entity lists from ``BasicEnricher`` are read from
                each record's ``"entities"`` field.
        """
        doc_node_id = f"doc:{document_id}"

        # Idempotency — skip if already indexed
        if self._graph.get_node(self._db, doc_node_id) is not None:
            return

        self._graph.add_node(
            self._db,
            "Document",
            {
                "document_id": document_id,
                "file_name": file_name,
                "document_type": document_type or "unknown",
            },
            node_id=doc_node_id,
        )

        for record in records:
            chunk_id = record.get("chunk_id", "")
            chunk_node_id = f"chunk:{chunk_id}"

            chunk_props: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text_preview": record.get("text", "")[:200],
            }
            if record.get("section_title"):
                chunk_props["section_title"] = record["section_title"]

            self._graph.add_node(self._db, "Chunk", chunk_props, node_id=chunk_node_id)
            self._graph.add_edge(self._db, doc_node_id, "HAS_CHUNK", chunk_node_id)

            entity_node_ids: List[str] = []
            for entity in record.get("entities", []):
                entity_text = entity.get("text", "").strip()
                entity_type = entity.get("type", "UNKNOWN")
                if not entity_text:
                    continue

                entity_node_id = f"entity:{entity_type}:{entity_text.lower()}"

                if self._graph.get_node(self._db, entity_node_id) is None:
                    self._graph.add_node(
                        self._db,
                        "Entity",
                        {"name": entity_text, "entity_type": entity_type},
                        node_id=entity_node_id,
                    )

                self._graph.add_edge(self._db, chunk_node_id, "MENTIONS", entity_node_id)
                entity_node_ids.append(entity_node_id)

            # CO_OCCURS_WITH edges between every pair of entities in this chunk
            for i, eid1 in enumerate(entity_node_ids):
                for eid2 in entity_node_ids[i + 1:]:
                    self._graph.add_edge(self._db, eid1, "CO_OCCURS_WITH", eid2)

    # ── delete ────────────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> None:
        """Remove the Document node and all its Chunk nodes from the graph.

        Entity nodes are retained because they may be shared with other
        documents. Only Document → Chunk subgraph is removed.
        """
        doc_node_id = f"doc:{document_id}"

        chunk_refs = self._graph.get_neighbors(
            self._db, doc_node_id, edge_type="HAS_CHUNK", direction="out"
        )
        for item in chunk_refs:
            node = item.get("node", {})
            chunk_node_id = node.get("id")
            if chunk_node_id:
                self._graph.delete_node(self._db, chunk_node_id)

        self._graph.delete_node(self._db, doc_node_id)

    # ── retrieval ─────────────────────────────────────────────────────────

    def graph_neighbors(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find chunks that mention *entity_name*, via graph traversal.

        Uses deterministic node-ID lookup (O(1) per entity type) rather than
        a full ``match_nodes`` scan, so it is fast even on large graphs.
        Matching is case-insensitive on the stored name.

        Args:
            entity_name: Entity text to search for.
            entity_type: Restrict to one entity type (PROPER_NOUN, DATE, …).
                When ``None``, all known entity types are tried.
            k: Maximum number of chunk results to return.

        Returns:
            List of dicts with keys: ``chunk_id``, ``document_id``,
            ``text_preview``, ``section_title`` (optional), ``entity``,
            ``entity_type``.
        """
        normalized = entity_name.lower()
        types_to_try = [entity_type] if entity_type else list(_KNOWN_ENTITY_TYPES)

        entity_node_ids: List[str] = []
        for etype in types_to_try:
            eid = f"entity:{etype}:{normalized}"
            if self._graph.get_node(self._db, eid) is not None:
                entity_node_ids.append(eid)

        results: List[Dict[str, Any]] = []
        seen: set = set()

        for entity_id in entity_node_ids:
            entity_node = self._graph.get_node(self._db, entity_id)
            entity_props = (entity_node or {}).get("properties", {})

            chunk_refs = self._graph.get_neighbors(
                self._db, entity_id, edge_type="MENTIONS", direction="in"
            )
            for item in chunk_refs:
                chunk_node = item.get("node", {})
                chunk_node_id = chunk_node.get("id", "")
                if chunk_node_id in seen:
                    continue
                seen.add(chunk_node_id)

                chunk_props = chunk_node.get("properties", {})
                result: Dict[str, Any] = {
                    "chunk_id": chunk_props.get("chunk_id", ""),
                    "document_id": chunk_props.get("document_id", ""),
                    "text_preview": chunk_props.get("text_preview", ""),
                    "entity": entity_props.get("name", entity_name),
                    "entity_type": entity_props.get("entity_type", ""),
                }
                if chunk_props.get("section_title"):
                    result["section_title"] = chunk_props["section_title"]
                results.append(result)

                if len(results) >= k:
                    return results

        return results
