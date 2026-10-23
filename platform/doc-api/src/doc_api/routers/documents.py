import os
from typing import Any, Dict, List, Optional

from blue_data.core.utils.path_utils import contain
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from doc_api.config import get_pipeline

router = APIRouter(prefix="/documents", tags=["documents"])


def _confine_path(path: str) -> str:
    """Confine a caller-supplied ingest/delete path to ``BLUE_DOC_ROOT``.

    The REST API is network-facing: without confinement a caller can name any
    server file the process can read, have it ingested, and read the text back via
    search (issue #75 / DOCSRC-07). When ``BLUE_DOC_ROOT`` is set, ``path`` must be
    a single filename that resolves strictly inside that directory — built with
    blue-data's ``contain()`` (no separators, no absolute paths, no ``..``, symlinks
    resolved), so an arbitrary server path is rejected with 400 (#126 / XR-08).

    When ``BLUE_DOC_ROOT`` is unset the path is returned unchanged — the endpoint
    still requires an API key (blue_data.server), so it is not reachable
    unauthenticated. Operators set ``BLUE_DOC_ROOT`` to lock ingestion to a root.
    """
    root = os.getenv("BLUE_DOC_ROOT")
    if not root:
        return path
    try:
        return contain(root, path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"path must be a filename within BLUE_DOC_ROOT: {exc}",
        )


# ── Request models ────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str
    collection: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    collection: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    mode: str = "vector"
    rich: bool = False
    context_chunks: int = 1


class EnsureIndexRequest(BaseModel):
    collection: Optional[str] = None


class DeleteRequest(BaseModel):
    path: Optional[str] = None
    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    collection: Optional[str] = None


class FindRequest(BaseModel):
    filter: Optional[Dict[str, Any]] = None


class FindThenSearchRequest(BaseModel):
    query: str
    find_filter: Optional[Dict[str, Any]] = None
    k: int = 5
    collection: Optional[str] = None
    mode: str = "vector"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ensure-index")
def ensure_index(request: EnsureIndexRequest):
    """Create the vector index for a collection. Idempotent — safe to call multiple times."""
    pipeline = get_pipeline()
    pipeline.ensure_index(collection=request.collection)
    return {"status": "ok", "collection": request.collection or pipeline.collection}


@router.post("/ingest")
def ingest(request: IngestRequest) -> Dict[str, Any]:
    """Parse, chunk, enrich, embed, and upsert a document.

    Supported file types: .txt, .pdf, .docx. Re-ingesting the same file is safe —
    content-hash deduplication means the same file always produces the same chunk IDs.

    Returns document_id and chunk_count. Store these if you need to delete the
    document later from a different process.
    """
    pipeline = get_pipeline()
    safe_path = _confine_path(request.path)
    try:
        records = pipeline.ingest(safe_path, collection=request.collection)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not records:
        raise HTTPException(status_code=422, detail="No chunks produced — file may be empty or unparseable")

    return {
        "document_id": records[0]["document_id"],
        "file_name": records[0]["file_name"],
        "chunk_count": len(records),
        "collection": request.collection or pipeline.collection,
    }


@router.post("/search")
def search(request: SearchRequest):
    """Search documents by semantic similarity.

    Returns a flat list of scored chunk dicts by default. Pass ``rich=true`` to
    get a structured response with provenance blocks, neighbor context, and
    by_document grouping — useful for citations and agent explainability.
    """
    pipeline = get_pipeline()
    try:
        results = pipeline.search(
            request.query,
            k=request.k,
            collection=request.collection,
            filters=request.filter,
            mode=request.mode,
            rich=request.rich,
            context_chunks=request.context_chunks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return results


@router.post("/find")
def find(request: FindRequest) -> List[Dict[str, Any]]:
    """Filter DocumentRecords by extracted fields — no embedding, no LLM.

    Filter syntax: ``{"document_type": "contract"}``,
    ``{"fields.total_due": {"gt": 30000}}``,
    ``{"fields.vendor": {"contains": "Acme"}}``.
    All conditions are ANDed. Pass an empty body to return all records.
    """
    pipeline = get_pipeline()
    return pipeline.find_records(filter=request.filter)


@router.post("/find-then-search")
def find_then_search(request: FindThenSearchRequest):
    """Filter documents by fields, then search text within matched documents.

    Combines DOCUMENT.FIND and DOCUMENT.SEARCH in one call — filters the
    schema store by ``find_filter``, then runs vector search scoped to the
    matched document IDs, and returns a merged ranked list.
    """
    pipeline = get_pipeline()
    try:
        # Thin adapter (ADR-012): the merge + mode-correct ranking lives in the
        # library, so REST never re-implements (and mis-sorts) it (#74).
        return pipeline.find_then_search(
            request.query,
            find_filter=request.find_filter,
            k=request.k,
            collection=request.collection,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/record/{document_id}")
def get_record(document_id: str) -> Dict[str, Any]:
    """Retrieve the DocumentRecord written at ingest time for a document."""
    pipeline = get_pipeline()
    record = pipeline.get_record(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record found for document_id: {document_id}")
    return record


@router.delete("/document")
def delete(request: DeleteRequest) -> Dict[str, Any]:
    """Delete all chunks for a document from the vector store.

    Two ways to identify the document:
    - Pass ``path``: works within the same server session.
    - Pass ``document_id`` + ``chunk_count``: works across sessions. Both values
      are returned by ``POST /documents/ingest``.
    """
    pipeline = get_pipeline()
    safe_path = _confine_path(request.path) if request.path else None
    try:
        deleted = pipeline.delete(
            path=safe_path,
            document_id=request.document_id,
            chunk_count=request.chunk_count,
            collection=request.collection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"deleted": deleted}
