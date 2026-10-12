"""MCP server exposing blue-data-doc-source as AI-callable tools.

Run:
    python platform/doc-mcp/src/server.py

Configure via environment variables:

    BLUE_DOC_EMBEDDING          openai | dummy  (default: dummy)
    OPENAI_API_KEY              required when BLUE_DOC_EMBEDDING=openai
    OPENAI_EMBEDDING_MODEL      default: text-embedding-3-small
    BLUE_DOC_REDIS_HOST         Redis host; omit to run without a vector store
    BLUE_DOC_REDIS_PORT         default: 6379
    BLUE_DOC_COLLECTION         default: documents
    BLUE_DOC_CHUNK_SIZE         default: 1000
    BLUE_DOC_OVERLAP            default: 150
    BLUE_DOC_OCR_FALLBACK       1 | true to enable Tesseract OCR on scanned PDFs
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
        OpenAIEmbeddingModel,
    )

    embedding_backend = os.getenv("BLUE_DOC_EMBEDDING", "dummy").lower()
    if embedding_backend == "openai":
        embedding_model = OpenAIEmbeddingModel(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        embedding_model = DummyEmbeddingModel()

    vector_source = None
    redis_host = os.getenv("BLUE_DOC_REDIS_HOST")
    if redis_host:
        from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
        vector_source = RedisVectorSource(
            name="redis",
            properties={"connection": {
                "host": redis_host,
                "port": int(os.getenv("BLUE_DOC_REDIS_PORT", "6379")),
                # Required by blue-data's create_index; also namespaces keys
                # (without it, keys are prefixed with the literal "None"). See #70.
                "scan_prefix": os.getenv("BLUE_DOC_REDIS_PREFIX", "bluedoc"),
            }},
        )

    ocr_fallback = os.getenv("BLUE_DOC_OCR_FALLBACK", "").lower() in ("1", "true", "yes")

    return DocumentPipeline(
        chunk_size=int(os.getenv("BLUE_DOC_CHUNK_SIZE", "1000")),
        overlap=int(os.getenv("BLUE_DOC_OVERLAP", "150")),
        embedding_model=embedding_model,
        vector_source=vector_source,
        enricher=BasicEnricher(),
        collection=os.getenv("BLUE_DOC_COLLECTION", "documents"),
        ocr_fallback=ocr_fallback,
    )


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def ingest_document(path: str, collection: str = "documents") -> str:
    """Ingest a document into the vector store.

    Parses the file, splits it into section-aware chunks, extracts entities and
    keywords, generates embeddings, and upserts to the configured vector store.
    Re-ingesting the same file is safe — content-hash deduplication means the
    same file always produces the same chunk IDs, so upsert overwrites rather
    than duplicates.

    Supported file types: .txt, .pdf, .docx

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
            "collection": collection,
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
    collection: str = "documents",
    k: int = 5,
) -> str:
    """Search documents by semantic similarity.

    Embeds the query and returns the k most relevant chunks from the vector
    store. Results include the chunk text, source file, section title, and page
    number (for PDFs). Use this to answer questions that require reading
    document content.

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
def ensure_index(collection: str = "documents") -> str:
    """Create the vector index for a collection.

    Must be called once before the first ingest when using a vector store.
    Safe to call multiple times — skips creation if the index already exists.
    Call this at the start of a session before ingesting any documents.
    """
    try:
        pipeline = _get_pipeline()
        pipeline.ensure_index(collection=collection)
        return json.dumps({"status": "ok", "collection": collection})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def delete_document(
    path: Optional[str] = None,
    document_id: Optional[str] = None,
    chunk_count: Optional[int] = None,
    collection: str = "documents",
) -> str:
    """Delete all chunks for a document from the vector store.

    Two ways to identify the document:
    - Pass path: works when the document was ingested in this server session.
    - Pass document_id and chunk_count: works across sessions. Both values are
      returned by ingest_document and can be persisted for later deletion.

    Returns the number of chunks successfully deleted.
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


if __name__ == "__main__":
    mcp.run()
