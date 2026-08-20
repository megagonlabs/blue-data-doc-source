from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from doc_api.config import get_pipeline

router = APIRouter(prefix="/documents", tags=["documents"])


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
    try:
        records = pipeline.ingest(request.path, collection=request.collection)
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
        matched = pipeline.find_records(filter=request.find_filter)
        if not matched:
            return []
        doc_ids = [r["document_id"] for r in matched]
        all_results = []
        for doc_id in doc_ids:
            hits = pipeline.search(
                request.query,
                k=request.k,
                collection=request.collection,
                filters={"document_id": doc_id},
                mode=request.mode,
            )
            all_results.extend(hits)
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return all_results[:request.k]
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
    try:
        deleted = pipeline.delete(
            path=request.path,
            document_id=request.document_id,
            chunk_count=request.chunk_count,
            collection=request.collection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"deleted": deleted}
