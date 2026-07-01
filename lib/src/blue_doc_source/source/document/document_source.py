from typing import Any, Dict, List, Optional

from blue_data.source.source import DataSource
from blue_data.source.vector.vector_source import VectorSource

from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .pipeline import DocumentPipeline


class DocumentDataSource(DataSource):
    """Top-level entry point for document ingestion and retrieval.

    Exposes an ``execute_query`` interface consistent with Blue's DataSource
    abstraction, dispatching on the ``"op"`` field of the request dict.

    Supported ops:

    - ``DOCUMENT.INGEST`` — parse, chunk, enrich, embed, and upsert a document
    - ``DOCUMENT.SEARCH`` — embed a query and return matching chunks via KNN

    Example::

        source = DocumentDataSource(
            name="contracts",
            embedding_model=OpenAIEmbeddingModel(),
            vector_source=RedisVectorSource(...),
            enricher=BasicEnricher(),
        )
        records = source.execute_query({"op": "DOCUMENT.INGEST", "path": "doc.pdf"})
        results = source.execute_query({"op": "DOCUMENT.SEARCH", "query": "eligibility", "k": 5})
    """

    def __init__(
        self,
        name: str = "document",
        embedding_model: Optional[EmbeddingModel] = None,
        vector_source: Optional[VectorSource] = None,
        enricher: Optional[Enricher] = None,
        chunk_size: int = 1000,
        overlap: int = 150,
        default_collection: str = "documents",
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            name: Source name registered in blue-data's DataRegistry.
            embedding_model: Embedding model for chunk and query vectors.
                Required for ``DOCUMENT.SEARCH`` and for upsert to fire on ingest.
            vector_source: VectorSource backend (e.g. RedisVectorSource from blue-data)
                for upsert and KNN search. Required for ``DOCUMENT.SEARCH``.
            enricher: Optional enrichment step run on each chunk during ingest.
            chunk_size: Maximum words per chunk.
            overlap: Word overlap between consecutive chunks.
            default_collection: Collection name used when the request does not
                specify one.
            properties: Optional metadata properties passed to DataSource.
        """
        super().__init__(name, properties)
        self._default_collection = default_collection
        self._pipeline = DocumentPipeline(
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
            vector_source=vector_source,
            enricher=enricher,
            collection=default_collection,
        )

    def execute_query(self, query, database=None, collection=None, optional_properties=None):
        op = query.get("op")

        if op == "DOCUMENT.INGEST":
            return self._ingest(query, collection)
        elif op == "DOCUMENT.SEARCH":
            return self._search(query, collection)
        else:
            raise ValueError(f"Unsupported op: {op!r}. Expected DOCUMENT.INGEST or DOCUMENT.SEARCH")

    def _ingest(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        path = request.get("path")
        if not path:
            raise ValueError("DOCUMENT.INGEST requires a 'path' field")

        collection = collection_override or request.get("collection")
        return self._pipeline.ingest(path, collection=collection)

    def _search(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        query_text = request.get("query")
        if not query_text:
            raise ValueError("DOCUMENT.SEARCH requires a 'query' field")

        collection = collection_override or request.get("collection")
        k = request.get("k", 5)
        return self._pipeline.search(query_text, k=k, collection=collection)
