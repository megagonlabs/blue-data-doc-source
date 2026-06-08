from typing import Any, Dict, List, Optional

from blue_doc_source.embeddings import EmbeddingModel
from blue_doc_source.enrichment import Enricher
from blue_doc_source.pipeline import DocumentPipeline
from blue_doc_source.vector_stores import VectorStore


class DocumentDataSource:
    """Top-level entry point for document ingestion and retrieval.

    Exposes an ``execute_query`` interface consistent with Blue's DataSource
    abstraction, dispatching on the ``"op"`` field of the request dict.

    Supported ops:

    - ``DOCUMENT.INGEST`` — parse, chunk, enrich, embed, and upsert a document
    - ``DOCUMENT.SEARCH`` — embed a query and return matching chunks via KNN

    Example::

        source = DocumentDataSource(
            embedding_model=OpenAIEmbeddingModel(),
            vector_store=RedisVectorAdapter(redis_source),
            enricher=BasicEnricher(),
        )
        records = source.execute_query({"op": "DOCUMENT.INGEST", "path": "doc.pdf"})
        results = source.execute_query({"op": "DOCUMENT.SEARCH", "query": "eligibility", "k": 5})
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        enricher: Optional[Enricher] = None,
        chunk_size: int = 1000,
        overlap: int = 150,
        default_collection: str = "documents",
    ):
        """
        Args:
            embedding_model: Embedding model for chunk and query vectors.
                Required for ``DOCUMENT.SEARCH`` and for upsert to fire on ingest.
            vector_store: Vector store backend for upsert and KNN search.
                Required for ``DOCUMENT.SEARCH``.
            enricher: Optional enrichment step run on each chunk during ingest.
            chunk_size: Maximum words per chunk.
            overlap: Word overlap between consecutive chunks.
            default_collection: Collection name used when the request does not
                specify one.
        """
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._enricher = enricher
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._default_collection = default_collection

    def execute_query(self, request: Dict[str, Any]) -> Any:
        op = request.get("op")

        if op == "DOCUMENT.INGEST":
            return self._ingest(request)
        elif op == "DOCUMENT.SEARCH":
            return self._search(request)
        else:
            raise ValueError(f"Unsupported op: {op!r}. Expected DOCUMENT.INGEST or DOCUMENT.SEARCH")

    def _ingest(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = request.get("path")
        if not path:
            raise ValueError("DOCUMENT.INGEST requires a 'path' field")

        collection = request.get("collection", self._default_collection)

        pipeline = DocumentPipeline(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            embedding_model=self._embedding_model,
            vector_store=self._vector_store,
            enricher=self._enricher,
            collection=collection,
        )

        return pipeline.ingest(path)

    def _search(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = request.get("query")
        if not query:
            raise ValueError("DOCUMENT.SEARCH requires a 'query' field")

        collection = request.get("collection", self._default_collection)
        k = request.get("k", 5)

        pipeline = DocumentPipeline(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            embedding_model=self._embedding_model,
            vector_store=self._vector_store,
            enricher=self._enricher,
            collection=collection,
        )

        return pipeline.search(query, k=k)
