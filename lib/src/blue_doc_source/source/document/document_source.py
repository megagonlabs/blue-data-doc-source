from typing import Any, Dict, List, Optional

from blue_data.source.source import DataSource
from blue_data.source.vector.vector_source import VectorSource

from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .nl_parser import NLQueryParser
from .pipeline import DocumentPipeline


class DocumentDataSource(DataSource):
    """Top-level entry point for document ingestion and retrieval.

    Exposes an ``execute_query`` interface consistent with Blue's DataSource
    abstraction, dispatching on the ``"op"`` field of the request dict.

    Supported ops:

    - ``DOCUMENT.INGEST`` — parse, chunk, enrich, embed, and upsert a document
    - ``DOCUMENT.SEARCH`` — embed a query and return matching chunks via KNN
    - ``DOCUMENT.ENSURE_INDEX`` — create the vector index (idempotent)
    - ``DOCUMENT.DELETE`` — remove all chunks for a document from the vector store

    Natural language queries are supported via ``execute_query_nl`` when an
    ``NLQueryParser`` is provided::

        source = DocumentDataSource(
            name="contracts",
            embedding_model=OpenAIEmbeddingModel(),
            vector_source=RedisVectorSource(...),
            nl_parser=NLQueryParser(),
        )
        results = source.execute_query_nl("find the termination clause")
        records = source.execute_query_nl("ingest /docs/policy.pdf into contracts")
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
        nl_parser: Optional[NLQueryParser] = None,
        properties: Optional[Dict[str, Any]] = None,
        ocr_fallback: bool = False,
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
            nl_parser: Optional ``NLQueryParser`` instance. Required for
                ``execute_query_nl``. Pass a custom ``parse_fn`` to use any
                LLM backend instead of OpenAI.
            properties: Optional metadata properties passed to DataSource.
            ocr_fallback: When ``True``, enables Tesseract OCR for scanned PDF
                pages with no text layer. Requires the ``[ocr]`` extras.
        """
        super().__init__(name, properties)
        self._default_collection = default_collection
        self._nl_parser = nl_parser
        self._pipeline = DocumentPipeline(
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
            vector_source=vector_source,
            enricher=enricher,
            collection=default_collection,
            ocr_fallback=ocr_fallback,
        )

    def execute_query(self, query, database=None, collection=None, optional_properties=None):
        op = query.get("op")

        if op == "DOCUMENT.INGEST":
            return self._ingest(query, collection)
        elif op == "DOCUMENT.SEARCH":
            return self._search(query, collection)
        elif op == "DOCUMENT.ENSURE_INDEX":
            return self._ensure_index(query, collection)
        elif op == "DOCUMENT.DELETE":
            return self._delete(query, collection)
        else:
            raise ValueError(
                f"Unsupported op: {op!r}. "
                "Expected DOCUMENT.INGEST, DOCUMENT.SEARCH, DOCUMENT.ENSURE_INDEX, or DOCUMENT.DELETE"
            )

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

    def _ensure_index(self, request: Dict[str, Any], collection_override=None) -> None:
        collection = collection_override or request.get("collection")
        self._pipeline.ensure_index(collection=collection)

    def execute_query_nl(self, text: str, database=None, collection=None, optional_properties=None):
        """Parse a natural language instruction and execute the corresponding op.

        Requires an ``NLQueryParser`` passed as ``nl_parser`` at construction time.

        Args:
            text: Natural language instruction, e.g.
                ``"find the termination clause in the contracts collection"``.

        Returns:
            The result of the dispatched ``execute_query`` call.

        Raises:
            ValueError: If no ``nl_parser`` was provided at construction.
        """
        if self._nl_parser is None:
            raise ValueError(
                "execute_query_nl requires an nl_parser. "
                "Pass NLQueryParser() as nl_parser= when constructing DocumentDataSource."
            )
        op_dict = self._nl_parser.parse(text)
        return self.execute_query(
            op_dict,
            database=database,
            collection=collection,
            optional_properties=optional_properties,
        )

    def _delete(self, request: Dict[str, Any], collection_override=None) -> int:
        path = request.get("path")
        document_id = request.get("document_id")
        chunk_count = request.get("chunk_count")
        collection = collection_override or request.get("collection")
        return self._pipeline.delete(
            path=path,
            document_id=document_id,
            chunk_count=chunk_count,
            collection=collection,
        )
