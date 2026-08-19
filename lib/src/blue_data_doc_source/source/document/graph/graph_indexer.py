from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from blue_data.source.graph.graph_source import GraphSource

# Entity types BasicEnricher can produce — used for deterministic ID lookup.
_KNOWN_ENTITY_TYPES = ("PROPER_NOUN", "DATE", "MONEY", "EMAIL", "PERCENT", "UNKNOWN")


class GraphIndexer:
    """Writes document/chunk/section/entity nodes and edges into a GraphSource.

    Graph schema
    ------------
    (Document {document_id, file_name, document_type, mime_type,
               ingested_at, chunk_count})
        -[:HAS_SECTION]->
    (Section {section_id, title, document_id, position})
        -[:HAS_CHUNK]->
    (Chunk {chunk_id, document_id, text_preview, chunk_index, page?,
            section_title?})
        -[:MENTIONS]->
    (Entity {name, entity_type})

    Additional edges:
        Document  -[:HAS_CHUNK]->       Chunk    (direct, regardless of section)
        Chunk     -[:NEXT]->            Chunk    (sequential forward)
        Chunk     -[:PREV]->            Chunk    (sequential backward)
        Entity    -[:CO_OCCURS_WITH]->  Entity   (pairs in same chunk)
        Entity    -[:APPEARS_IN]->      Document (direct doc-level presence)

    Node IDs are deterministic:
        Document → ``doc:{document_id}``
        Section  → ``section:{document_id}:{title.lower()}``
        Chunk    → ``chunk:{chunk_id}``
        Entity   → ``entity:{entity_type}:{name.lower()}``

    Re-ingesting the exact same file is a no-op (Document node already exists).
    Entity nodes are shared across documents and are never removed by
    ``delete_document``. Section nodes belong to one document and are removed
    alongside the document on delete.
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
                pipeline. Reads ``entities``, ``section_title``, ``page_start``,
                ``mime_type``, and ``metadata.ingested_at`` from each record.
        """
        doc_node_id = f"doc:{document_id}"

        # Idempotency — skip if already indexed
        if self._graph.get_node(self._db, doc_node_id) is not None:
            return

        first = records[0] if records else {}
        self._graph.add_node(
            self._db,
            "Document",
            {
                "document_id": document_id,
                "file_name": file_name,
                "document_type": document_type or "unknown",
                "mime_type": first.get("mime_type", "text/plain"),
                "ingested_at": first.get("metadata", {}).get("ingested_at", ""),
                "chunk_count": len(records),
            },
            node_id=doc_node_id,
        )

        # section_title → section_node_id  (tracks per-document section nodes)
        seen_sections: Dict[str, str] = {}
        chunk_node_ids: List[str] = []
        doc_entity_ids: Set[str] = set()

        for chunk_index, record in enumerate(records):
            chunk_id = record.get("chunk_id", "")
            chunk_node_id = f"chunk:{chunk_id}"

            chunk_props: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text_preview": record.get("text", "")[:200],
                "chunk_index": chunk_index,
            }
            if record.get("page_start") is not None:
                chunk_props["page"] = record["page_start"]
            if record.get("section_title"):
                chunk_props["section_title"] = record["section_title"]

            self._graph.add_node(self._db, "Chunk", chunk_props, node_id=chunk_node_id)
            self._graph.add_edge(self._db, doc_node_id, "HAS_CHUNK", chunk_node_id)
            chunk_node_ids.append(chunk_node_id)

            # Section node — one per distinct section_title per document
            section_title = record.get("section_title")
            if section_title:
                section_node_id = seen_sections.get(section_title)
                if section_node_id is None:
                    section_node_id = f"section:{document_id}:{section_title.lower()}"
                    self._graph.add_node(
                        self._db,
                        "Section",
                        {
                            "section_id": section_node_id,
                            "title": section_title,
                            "document_id": document_id,
                            "position": chunk_index,
                        },
                        node_id=section_node_id,
                    )
                    self._graph.add_edge(self._db, doc_node_id, "HAS_SECTION", section_node_id)
                    seen_sections[section_title] = section_node_id
                self._graph.add_edge(self._db, section_node_id, "HAS_CHUNK", chunk_node_id)

            # Entity nodes and Chunk -[:MENTIONS]-> Entity edges
            chunk_entity_ids: List[str] = []
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
                chunk_entity_ids.append(entity_node_id)
                doc_entity_ids.add(entity_node_id)

            # CO_OCCURS_WITH edges between every pair of entities in this chunk
            for i, eid1 in enumerate(chunk_entity_ids):
                for eid2 in chunk_entity_ids[i + 1:]:
                    self._graph.add_edge(self._db, eid1, "CO_OCCURS_WITH", eid2)

        # NEXT / PREV sequential adjacency edges between consecutive chunks
        for i in range(len(chunk_node_ids) - 1):
            self._graph.add_edge(self._db, chunk_node_ids[i], "NEXT", chunk_node_ids[i + 1])
            self._graph.add_edge(self._db, chunk_node_ids[i + 1], "PREV", chunk_node_ids[i])

        # Entity -[:APPEARS_IN]-> Document (direct document-level presence)
        for entity_node_id in doc_entity_ids:
            self._graph.add_edge(self._db, entity_node_id, "APPEARS_IN", doc_node_id)

    # ── delete ────────────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> None:
        """Remove the Document, its Section nodes, and its Chunk nodes.

        Entity nodes are retained because they may be shared with other
        documents. Edges attached to removed nodes are also cleaned up by
        the backend's delete_node (DETACH DELETE on Neo4j; incident-edge
        removal on MemoryGraphSource).
        """
        doc_node_id = f"doc:{document_id}"

        # Delete Section nodes first
        for item in self._graph.get_neighbors(
            self._db, doc_node_id, edge_type="HAS_SECTION", direction="out"
        ):
            nid = item.get("node", {}).get("id")
            if nid:
                self._graph.delete_node(self._db, nid)

        # Delete Chunk nodes
        for item in self._graph.get_neighbors(
            self._db, doc_node_id, edge_type="HAS_CHUNK", direction="out"
        ):
            nid = item.get("node", {}).get("id")
            if nid:
                self._graph.delete_node(self._db, nid)

        self._graph.delete_node(self._db, doc_node_id)

    # ── traversal ─────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse the graph from *node_id* and return neighbouring nodes.

        Thin wrapper over ``GraphSource.get_neighbors`` that normalises the
        result into a flat dict list suitable for the ``DOCUMENT.GRAPH_QUERY``
        op response.

        Args:
            node_id: Graph node ID to start from (e.g. ``doc:doc_abc``).
            edge_type: Edge type to follow (e.g. ``HAS_CHUNK``, ``NEXT``).
                When ``None``, all edge types are traversed.
            direction: ``"out"``, ``"in"``, or ``"both"`` (default).
            k: Maximum number of results. ``None`` returns all.

        Returns:
            List of dicts with ``node_id``, ``label``, ``properties``,
            ``edge_type``, and ``direction``.
        """
        raw = self._graph.get_neighbors(
            self._db, node_id, edge_type=edge_type, direction=direction
        )
        results: List[Dict[str, Any]] = []
        for item in raw:
            node = item.get("node", {})
            edge = item.get("edge", {})
            results.append({
                "node_id": node.get("id", ""),
                "label": node.get("label", ""),
                "properties": node.get("properties", {}),
                "edge_type": edge.get("edge_type", edge_type or ""),
                "direction": item.get("direction", direction),
            })
            if k is not None and len(results) >= k:
                break
        return results

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

            for item in self._graph.get_neighbors(
                self._db, entity_id, edge_type="MENTIONS", direction="in"
            ):
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
