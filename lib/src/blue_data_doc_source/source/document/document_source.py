from pathlib import Path
from typing import Any, Dict, List, Optional

from blue_data.core.schema import DataSchema
from blue_data.source.source import DataSource
from blue_data.source.vector.vector_source import VectorSource

from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .nl_parser import NLQueryParser
from .pipeline import DocumentPipeline
from .schema.base import DocumentClassifier, SchemaExtractor


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
        classifier: Optional[DocumentClassifier] = None,
        schema_extractor: Optional[SchemaExtractor] = None,
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
        self.properties["connection"]["protocol"] = "document"
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
            classifier=classifier,
            schema_extractor=schema_extractor,
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
        elif op == "DOCUMENT.GET_SCHEMA":
            return self._get_schema(query)
        elif op == "DOCUMENT.SEARCH_RICH":
            return self._search_rich(query, collection)
        else:
            raise ValueError(
                f"Unsupported op: {op!r}. "
                "Expected DOCUMENT.INGEST, DOCUMENT.SEARCH, DOCUMENT.ENSURE_INDEX, "
                "DOCUMENT.DELETE, DOCUMENT.GET_SCHEMA, or DOCUMENT.SEARCH_RICH"
            )

    _PROTOCOL_BY_SUFFIX = {
        ".txt": "document/txt",
        ".pdf": "document/pdf",
        ".docx": "document/docx",
    }

    def _ingest(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        path = request.get("path")
        if not path:
            raise ValueError("DOCUMENT.INGEST requires a 'path' field")

        suffix = Path(path).suffix.lower()
        self.properties["connection"]["protocol"] = self._PROTOCOL_BY_SUFFIX.get(suffix, "document")

        collection = collection_override or request.get("collection")
        return self._pipeline.ingest(path, collection=collection)

    def _search(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        query_text = request.get("query")
        if not query_text:
            raise ValueError("DOCUMENT.SEARCH requires a 'query' field")

        collection = collection_override or request.get("collection")
        k = request.get("k", 5)
        filters = request.get("filter")
        return self._pipeline.search(query_text, k=k, collection=collection, filters=filters)

    def _ensure_index(self, request: Dict[str, Any], collection_override=None) -> None:
        collection = collection_override or request.get("collection")
        self._pipeline.ensure_index(collection=collection)

    def execute_query_nl(self, query: str, database=None, collection=None, schema=None, llm_fn=None, optional_properties=None):
        """Parse a natural language instruction and execute the corresponding op.

        Requires an ``NLQueryParser`` passed as ``nl_parser`` at construction time.
        The ``schema`` and ``llm_fn`` parameters are accepted to satisfy the
        ``DataSource`` ABC contract but are not used — doc-source uses its own
        ``NLQueryParser`` for intent extraction.

        Args:
            query: Natural language instruction, e.g.
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
        op_dict = self._nl_parser.parse(query)
        return self.execute_query(
            op_dict,
            database=database,
            collection=collection,
            optional_properties=optional_properties,
        )

    def fetch_databases(self) -> List[str]:
        """Return the logical database namespace for this source."""
        return ["documents"]

    def fetch_database_collections(self, database: str) -> List[str]:
        """Return the collections known to this source.

        The ``database`` argument is accepted for API compatibility; all
        collections belong to the single ``"documents"`` database.
        """
        return [self._default_collection]

    def fetch_database_collection_entities(self, database: str, collection: str) -> dict:
        """Return a DataSchema entity map describing the ChunkRecord schema.

        DataRegistry's ``sync_source`` calls this to build the registry
        hierarchy. Returning the ChunkRecord fields as attributes lets the
        registry display what data lives in this source without requiring an
        actual live query.
        """
        schema = DataSchema()
        schema.add_entity("ChunkRecord", description="A parsed and embedded document chunk")
        attrs = [
            ("document_id",   "str",       "SHA-256 content-addressed document identifier"),
            ("chunk_id",      "str",       "Unique chunk: {document_id}_chunk_{index:06d}"),
            ("text",          "str",       "Raw chunk text"),
            ("file_name",     "str",       "Source file basename"),
            ("source_type",   "str",       "MIME type of the source file"),
            ("mime_type",     "str",       "MIME type (same as source_type)"),
            ("page_start",    "int|None",  "First page of chunk (1-based; None for .txt/.docx)"),
            ("page_end",      "int|None",  "Last page of chunk (always == page_start for PDFs)"),
            ("section_title", "str|None",  "Nearest heading above this chunk"),
            ("document_type", "str|None",  "Classified type: contract, invoice, policy, research_paper, financial_report, unknown"),
            ("content_type",  "str|None",  "'text' or 'table' (PlumberPDFParser only)"),
            ("entities",      "list[dict]","Named-entity spans from BasicEnricher"),
            ("semantic_hints","dict",      "word_count, keywords, and other enrichment hints"),
        ]
        for name, typ, desc in attrs:
            schema.add_entity_attribute("ChunkRecord", name, typ, desc)
        return schema.get_entities()

    def _search_rich(self, request: Dict[str, Any], collection_override=None) -> Dict[str, Any]:
        query_text = request.get("query")
        if not query_text:
            raise ValueError("DOCUMENT.SEARCH_RICH requires a 'query' field")
        collection = collection_override or request.get("collection")
        k = request.get("k", 5)
        filters = request.get("filter")
        context_chunks = request.get("context_chunks", 1)
        return self._pipeline.search_rich(
            query_text,
            k=k,
            collection=collection,
            filters=filters,
            context_chunks=context_chunks,
        )

    def _get_schema(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        document_id = request.get("document_id")
        if not document_id:
            raise ValueError("DOCUMENT.GET_SCHEMA requires a 'document_id' field")
        schema = self._pipeline.get_schema(document_id)
        if schema is None:
            raise ValueError(
                f"No schema found for document_id={document_id!r}. "
                "Schema is populated during ingest when a classifier is configured."
            )
        return schema

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
