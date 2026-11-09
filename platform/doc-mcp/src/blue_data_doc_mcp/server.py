"""MCP server exposing blue-data-doc-source as AI-callable tools.

Run (after ``pip install`` of this package):
    blue-data-doc-mcp
or, from a source checkout:
    python -m blue_data_doc_mcp.server

Configure via environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    OPENAI_EMBEDDING_MODEL      default: text-embedding-3-small
    BLUE_DOC_REDIS_URL          full redis://|rediss:// URL (host, auth, TLS, db); or use:
    BLUE_DOC_REDIS_HOST         Redis host; omit (and no URL) to run without a vector store
    BLUE_DOC_REDIS_PORT         default: 6379
    BLUE_DOC_REDIS_USERNAME     Redis ACL username (optional)
    BLUE_DOC_REDIS_PASSWORD     Redis password (optional)
    BLUE_DOC_REDIS_SSL          1 | true to connect over TLS
    BLUE_DOC_REDIS_PREFIX       key namespace / scan prefix; default: bluedoc
    BLUE_DOC_COLLECTION         default: documents
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR on scanned PDFs
    BLUE_DOC_GRAPH              1 | true to enable the entity graph (graph_neighbors / graph_query tools)
"""
import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("blue-doc-source")

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline = None


def _build_pipeline():
    from blue_data_doc_source import (
        BasicEnricher,
        DocumentPipeline,
        DummyEmbeddingModel,
        KeywordClassifier,
        OpenAIEmbeddingModel,
    )

    embedding_backend = os.getenv("BLUE_DOC_EMBEDDING", "dummy").lower()
    if embedding_backend == "openai":
        embedding_model = OpenAIEmbeddingModel(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        embedding_model = DummyEmbeddingModel()

    # Forward the whole Redis connection — URL, host/port, username, password, TLS — so a
    # password-protected or TLS Redis works, not just host/port (#195).
    from blue_data_doc_source import record_store_from_env, redis_vector_source_from_env
    vector_source = redis_vector_source_from_env("BLUE_DOC")

    # Optional entity/structure graph for the graph tools (in-process; opt-in).
    graph_source = None
    if os.getenv("BLUE_DOC_GRAPH", "").lower() in ("1", "true", "yes"):
        from blue_data.source.graph.memory_graph_source import MemoryGraphSource
        graph_source = MemoryGraphSource("documents")

    ocr_fallback = os.getenv("BLUE_DOC_OCR_FALLBACK", "").lower() in ("1", "true", "yes")

    return DocumentPipeline(
        chunk_size=int(os.getenv("BLUE_DOC_CHUNK_SIZE", "1000")),
        overlap=int(os.getenv("BLUE_DOC_OVERLAP", "150")),
        embedding_model=embedding_model,
        vector_source=vector_source,
        enricher=BasicEnricher(),
        # Zero-API-cost classification so document_type + records exist, which the
        # find_documents / get_document_record tools query (#66).
        classifier=KeywordClassifier(),
        # Persist records (Redis when configured, else a file) so find / record / path-
        # delete survive the next MCP session instead of being lost with the process (#188).
        record_store=record_store_from_env("BLUE_DOC"),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
        graph_source=graph_source,
    )


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def ingest_document(path: str, collection: Optional[str] = None) -> str:
    """Ingest a document into the vector store.

    Parses the file, splits it into section-aware chunks, extracts entities and
    keywords, generates embeddings, and upserts to the configured vector store.
    Re-ingesting the same file is safe — content-hash deduplication means the
    same file always produces the same chunk IDs, so upsert overwrites rather
    than duplicates.

    Supported file types: .txt, .pdf, .docx

    When ``collection`` is omitted it defaults to the server's configured
    collection (``BLUE_DOC_COLLECTION``), not a hard-coded ``"documents"`` (#151).

    Returns document_id and chunk_count. Store these values if you need to
    delete the document later in a different session.
    """
    try:
        pipeline = _get_pipeline()
        records = pipeline.ingest(path, collection=collection)
        if not records:
            return json.dumps({"error": "No chunks produced — file may be empty or unparseable"})
        return json.dumps({
            "document_id": records[0]["document_id"],
            "chunk_count": len(records),
            "file_name": records[0]["file_name"],
            "collection": collection or pipeline.collection,
        })
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Ingest failed: {exc}"})


@mcp.tool()
def search_documents(
    query: str,
    collection: Optional[str] = None,
    k: int = 5,
) -> str:
    """Search documents by semantic similarity.

    Embeds the query and returns the k most relevant chunks from the vector
    store. Results include the chunk text, source file, section title, and page
    number (for PDFs). Use this to answer questions that require reading
    document content.

    When ``collection`` is omitted it defaults to the server's configured
    collection (``BLUE_DOC_COLLECTION``), not a hard-coded ``"documents"`` (#151).

    Requires BLUE_DOC_REDIS_HOST to be set and the collection to have been
    indexed via ensure_index before ingestion.
    """
    try:
        pipeline = _get_pipeline()
        results = pipeline.search(query, k=k, collection=collection)
        hits = []
        for r in results:
            hit: dict = {
                "chunk_id": r.get("chunk_id"),
                "text": r.get("text"),
                "file_name": r.get("file_name"),
            }
            if r.get("section_title"):
                hit["section_title"] = r["section_title"]
            if r.get("page_start") is not None:
                hit["page"] = r["page_start"]
            hits.append(hit)
        return json.dumps({"results": hits, "count": len(hits)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Search failed: {exc}"})


@mcp.tool()
def ensure_index(collection: Optional[str] = None) -> str:
    """Create the vector index for a collection.

    Must be called once before the first ingest when using a vector store.
    Safe to call multiple times — skips creation if the index already exists.
    Call this at the start of a session before ingesting any documents.

    When ``collection`` is omitted it defaults to the server's configured
    collection (``BLUE_DOC_COLLECTION``), not a hard-coded ``"documents"`` (#151).
    """
    try:
        pipeline = _get_pipeline()
        pipeline.ensure_index(collection=collection)
        return json.dumps({"status": "ok", "collection": collection or pipeline.collection})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def delete_document(
    path: Optional[str] = None,
    document_id: Optional[str] = None,
    chunk_count: Optional[int] = None,
    collection: Optional[str] = None,
) -> str:
    """Delete all chunks for a document from the vector store.

    Two ways to identify the document:
    - Pass path: works when the document was ingested in this server session. The
      document's own ingested collection is used — ``collection`` is ignored.
    - Pass document_id and chunk_count: works across sessions. Both values are
      returned by ingest_document and can be persisted for later deletion.

    ``collection`` defaults to None (the server's default collection) rather than a
    hard-coded "documents", so a cross-session delete isn't misdirected (#78).

    Returns the number of chunks actually removed (0 if the document was not found).
    """
    try:
        pipeline = _get_pipeline()
        deleted = pipeline.delete(
            path=path,
            document_id=document_id,
            chunk_count=chunk_count,
            collection=collection,
        )
        return json.dumps({"deleted": deleted})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Delete failed: {exc}"})


@mcp.tool()
def find_documents(
    filter: Optional[dict] = None,
    limit: int = 50,
) -> str:
    """Filter documents by their extracted fields — no embedding, no LLM.

    Deterministic structured query over the per-document records written at
    ingest (document_type + extracted fields). Use this to answer questions a
    vector search cannot, e.g. "which contracts are governed by California" or
    "invoices over $30,000".

    Filter syntax (all conditions ANDed):
      {"document_type": "contract"}
      {"fields.total_due": {"gt": 30000}}
      {"fields.vendor": {"contains": "Acme"}}

    Returns up to `limit` matching document records.
    """
    try:
        pipeline = _get_pipeline()
        records = pipeline.find_records(filter)
        return json.dumps({"records": records[:limit], "count": min(len(records), limit)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Find failed: {exc}"})


@mcp.tool()
def find_then_search(
    query: str,
    find_filter: Optional[dict] = None,
    collection: Optional[str] = None,
    k: int = 5,
) -> str:
    """Filter documents by fields, then semantic-search text within the matches.

    Combines a structured filter (DOCUMENT.FIND) with text retrieval
    (DOCUMENT.SEARCH): narrows the corpus to documents matching `find_filter`,
    searches `query` within each, and returns a merged ranked list of chunks.
    Use for "find <text> in documents where <field condition>".

    When ``collection`` is omitted it defaults to the server's configured
    collection (``BLUE_DOC_COLLECTION``), not a hard-coded ``"documents"`` (#151).
    """
    try:
        pipeline = _get_pipeline()
        # Call the shared merge (correctly ranked — ascending cosine distance in
        # vector mode) instead of re-implementing it here. The duplicated loop sorted
        # distances descending, returning the worst match first (#147 / ADR-012).
        hits = pipeline.find_then_search(query, find_filter=find_filter, k=k, collection=collection)
        results = [
            {key: h.get(key) for key in ("chunk_id", "text", "file_name", "document_type", "score")}
            for h in hits
        ]
        return json.dumps({"results": results, "count": len(results)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Find-then-search failed: {exc}"})


@mcp.tool()
def get_document_record(document_id: str) -> str:
    """Retrieve the structured record for a document.

    Returns the document_type, classifier confidence, and extracted fields
    written at ingest time. `document_id` is the content-hash ID returned by
    ingest_document.
    """
    try:
        pipeline = _get_pipeline()
        record = pipeline.get_record(document_id)
        if record is None:
            return json.dumps({"error": f"No record found for document_id: {document_id}"})
        return json.dumps(record)
    except Exception as exc:
        return json.dumps({"error": f"Get record failed: {exc}"})


@mcp.tool()
def graph_neighbors(
    entity: str,
    entity_type: Optional[str] = None,
    k: int = 10,
) -> str:
    """Find chunks that mention a named entity, via the entity graph.

    Exact entity lookup by deterministic node ID — every mention, no embeddings,
    no corpus scan. Requires the graph backend (start the server with
    BLUE_DOC_GRAPH=1).
    """
    try:
        pipeline = _get_pipeline()
        neighbors = pipeline.graph_neighbors(entity, entity_type=entity_type, k=k)
        return json.dumps({"neighbors": neighbors, "count": len(neighbors)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Graph neighbors failed: {exc}"})


@mcp.tool()
def graph_query(
    node_id: str,
    edge_type: Optional[str] = None,
    direction: str = "both",
    k: Optional[int] = None,
) -> str:
    """Traverse the entity/structure graph from any node.

    Start from a document (`doc:...`), chunk (`chunk:...`), section, or entity
    (`entity:PROPER_NOUN:acme corp`) node and walk edges (HAS_CHUNK, NEXT/PREV,
    MENTIONS, APPEARS_IN, CO_OCCURS_WITH). Requires BLUE_DOC_GRAPH=1.
    """
    try:
        pipeline = _get_pipeline()
        nodes = pipeline.graph_query(node_id, edge_type=edge_type, direction=direction, k=k)
        return json.dumps({"nodes": nodes, "count": len(nodes)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Graph query failed: {exc}"})


def main():
    """Console-script entry point (``blue-data-doc-mcp``). Starts the server."""
    mcp.run()


if __name__ == "__main__":
    main()
