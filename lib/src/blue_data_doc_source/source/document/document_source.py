from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from blue_data.core.schema import DataSchema
from blue_data.core.utils.path_utils import contain
from blue_data.source.source import DataSource
from blue_data.source.vector.vector_source import VectorSource

from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .nl_parser import NLQueryParser
from .pipeline import DocumentPipeline
from .schema.base import DocumentClassifier, FieldExtractor
from .schema.schema_store import RecordStore

if TYPE_CHECKING:
    from blue_data.source.graph.graph_source import GraphSource


# Ops a natural-language query may trigger by default. State-changing ops
# (INGEST, DELETE, ENSURE_INDEX) are excluded so that untrusted text routed
# through execute_query_nl — an end-user question, or text an agent forwards
# from a retrieved document — cannot mutate state. Opt in with
# ``allow_write_ops=True``. See issue #79.
_READ_ONLY_NL_OPS = frozenset({
    "DOCUMENT.SEARCH",
    "DOCUMENT.FIND",
    "DOCUMENT.FIND_THEN_SEARCH",
    "DOCUMENT.GET_RECORD",
    "DOCUMENT.GRAPH_NEIGHBORS",
    "DOCUMENT.GRAPH_QUERY",
})


class DocumentDataSource(DataSource):
    """Top-level entry point for document ingestion and retrieval.

    Extends blue-data's ``DataSource`` ABC and registers in the same
    ``DataRegistry`` as SQL, graph, and JSON sources. Exposes an
    ``execute_query`` interface that dispatches on the ``"op"`` field of the
    request dict.

    Supported ops:

    - ``DOCUMENT.INGEST`` — parse, chunk, enrich, embed, and upsert a document
    - ``DOCUMENT.SEARCH`` — embed a query and return matching chunks; set ``"rich": true`` for provenance, neighbor context, and ``by_document`` grouping
    - ``DOCUMENT.ENSURE_INDEX`` — create the vector index (idempotent)
    - ``DOCUMENT.DELETE`` — remove all chunks for a document from the vector store
    - ``DOCUMENT.GET_RECORD`` — retrieve the DocumentRecord extracted at ingest time
    - ``DOCUMENT.FIND`` — filter DocumentRecords by extracted field values (no embedding, deterministic)
    - ``DOCUMENT.FIND_THEN_SEARCH`` — filter DocumentRecords by field values, then search text within matched documents
    - ``DOCUMENT.GRAPH_NEIGHBORS`` — find chunks mentioning a named entity via graph traversal (requires ``graph_source``)
    - ``DOCUMENT.GRAPH_QUERY`` — general graph traversal from any node ID; returns neighbouring nodes (requires ``graph_source``)

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
        field_extractor: Optional[FieldExtractor] = None,
        record_store: Optional[RecordStore] = None,
        graph_source: Optional["GraphSource"] = None,
        graph_database: str = "documents",
        documents_root: Optional[str] = None,
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
            classifier: Optional document type classifier run once per document
                at ingest time. When set, ``document_type`` is propagated to
                every chunk and schema records are written to ``record_store``.
            field_extractor: Optional field extractor run after classification.
                Makes one LLM call per document to populate ``fields`` (parties,
                governing_law, total_due, etc.) in the schema record.
            record_store: Backend for schema records. Defaults to
                ``MemoryRecordStore`` (in-process). Pass ``FileRecordStore(path)``
                to persist records across restarts.
            graph_source: Optional blue-data ``GraphSource`` backend. When set,
                every ingested document is indexed into the graph and
                ``DOCUMENT.GRAPH_NEIGHBORS`` queries are enabled.
            graph_database: Graph database name. Defaults to ``"documents"``.
        """
        super().__init__(name, properties)
        self.properties.setdefault("connection", {})["protocol"] = "document"
        props = self.properties

        # DataSource contract §4 (blue-data design/14): a source must be
        # reconstructible from (name, properties) alone — the registry's
        # connect_source rebuilds it from the stored record, with no Python-object
        # collaborators to hand in. So the search-critical collaborators (embedding
        # model, vector store, enricher) are built from ``properties`` here. A
        # collaborator passed explicitly still wins (direct, in-process use);
        # otherwise it is built from the matching properties key when present;
        # otherwise it stays unset (the historical no-properties default, so existing
        # callers are unaffected). Scalars take the properties value when set, else
        # the keyword default — so a rebuilt source reproduces the same config.
        # classifier / field_extractor / record_store / graph_source remain
        # object-injected advanced features (not needed for registry-hosted search).
        default_collection = props.get("collection", default_collection)
        chunk_size = props.get("chunk_size", chunk_size)
        overlap = props.get("overlap", overlap)
        graph_database = props.get("graph_database", graph_database)
        ocr_fallback = props.get("ocr_fallback", ocr_fallback)
        # Optional documents root: when set, caller-supplied ingest paths are
        # resolved under it and anything outside is refused (#126). None keeps the
        # historical unconfined behaviour.
        self._documents_root = props.get("documents_root", documents_root)

        if embedding_model is None:
            embedding_model = self._build_embedding(props)
        if vector_source is None:
            vector_source = self._build_vector_source(props)
        if enricher is None:
            enricher = self._build_enricher(props)

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
            field_extractor=field_extractor,
            record_store=record_store,
            graph_source=graph_source,
            graph_database=graph_database,
        )

    # ---------------------------------------------------------------- builders
    # Build collaborators from a ``properties`` dict so the source is
    # reconstructible from its stored record (DataSource contract §4). Each returns
    # ``None`` when its backend key is absent, preserving the historical
    # "no properties → no collaborator" constructor default.

    @staticmethod
    def _build_embedding(props: Dict[str, Any]) -> Optional[EmbeddingModel]:
        backend = props.get("embedding")
        if not backend:
            return None
        backend = str(backend).lower()
        if backend == "openai":
            from .embeddings.openai_embedding import OpenAIEmbeddingModel
            return OpenAIEmbeddingModel(
                model=props.get("openai_embedding_model", "text-embedding-3-small")
            )
        if backend == "dummy":
            from .embeddings.dummy_embedding import DummyEmbeddingModel
            return DummyEmbeddingModel()
        raise ValueError(f"Unknown embedding backend {backend!r} (expected 'openai' or 'dummy')")

    @staticmethod
    def _build_vector_source(props: Dict[str, Any]) -> Optional[VectorSource]:
        backend = props.get("vector")
        if not backend:
            return None
        backend = str(backend).lower()
        if backend == "memory":
            from blue_data.source.vector.memory_vector_source import MemoryVectorSource
            return MemoryVectorSource(name="memory")
        if backend == "redis":
            from blue_data.source.vector.redis.redis_vector_source import RedisVectorSource
            conn = {k: v for k, v in props.get("connection", {}).items() if k != "protocol"}
            return RedisVectorSource(name="redis", properties={"connection": {
                "host": conn.get("host", "localhost"),
                "port": int(conn.get("port", 6379)),
                "scan_prefix": conn.get("scan_prefix", "documents:"),
            }})
        raise ValueError(f"Unknown vector backend {backend!r} (expected 'memory' or 'redis')")

    @staticmethod
    def _build_enricher(props: Dict[str, Any]) -> Optional[Enricher]:
        backend = props.get("enricher")
        if not backend:
            return None
        backend = str(backend).lower()
        if backend == "none":
            return None
        if backend == "basic":
            from .enrichment import BasicEnricher
            return BasicEnricher()
        if backend == "llm":
            from .enrichment import BasicEnricher, LLMEnricher
            llm_model = props.get("llm_model", "gpt-4o-mini")

            def _llm_fn(prompt: str) -> str:
                from openai import OpenAI
                return OpenAI().chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                ).choices[0].message.content or ""

            return LLMEnricher(llm_fn=_llm_fn, fallback=BasicEnricher())
        raise ValueError(f"Unknown enricher backend {backend!r} (expected 'none', 'basic' or 'llm')")

    def execute_query(self, query, database=None, collection=None, optional_properties=None):
        """Dispatch a structured op request to the appropriate pipeline method.

        Args:
            query: Dict with an ``"op"`` key and op-specific fields. See class
                docstring for the full list of supported ops and their required
                fields.
            collection: Overrides ``query["collection"]`` when provided.

        Returns:
            Op-specific result: ``list[dict]`` for INGEST/SEARCH/FIND,
            ``dict`` for SEARCH (when ``rich=true``)/GET_RECORD, ``None`` for
            ENSURE_INDEX, ``int`` (deleted count) for DELETE.

        Raises:
            ValueError: For unsupported ops or missing required fields.
        """
        op = query.get("op")

        if op == "DOCUMENT.INGEST":
            return self._ingest(query, collection)
        elif op == "DOCUMENT.SEARCH":
            return self._search(query, collection)
        elif op == "DOCUMENT.ENSURE_INDEX":
            return self._ensure_index(query, collection)
        elif op == "DOCUMENT.DELETE":
            return self._delete(query, collection)
        elif op == "DOCUMENT.GET_RECORD":
            return self._get_record(query)
        elif op == "DOCUMENT.FIND":
            return self._find(query)
        elif op == "DOCUMENT.FIND_THEN_SEARCH":
            return self._find_then_search(query, collection)
        elif op == "DOCUMENT.GRAPH_NEIGHBORS":
            return self._graph_neighbors(query)
        elif op == "DOCUMENT.GRAPH_QUERY":
            return self._graph_query(query)
        else:
            raise ValueError(
                f"Unsupported op: {op!r}. "
                "Expected DOCUMENT.INGEST, DOCUMENT.SEARCH, DOCUMENT.ENSURE_INDEX, "
                "DOCUMENT.DELETE, DOCUMENT.GET_RECORD, DOCUMENT.FIND, "
                "DOCUMENT.FIND_THEN_SEARCH, DOCUMENT.GRAPH_NEIGHBORS, or DOCUMENT.GRAPH_QUERY"
            )

    _PROTOCOL_BY_SUFFIX = {
        ".txt": "document/txt",
        ".pdf": "document/pdf",
        ".docx": "document/docx",
    }

    def ingest_text(
        self,
        text: str,
        file_name: str,
        mime_type: str = "text/plain",
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk, enrich, and embed pre-extracted text, bypassing file parsing and execute_query.

        Thin public wrapper around ``DocumentPipeline.ingest_text`` for callers
        that already have text in hand (e.g. a transcript) and don't want to
        go through the ``DOCUMENT.INGEST`` op, which only accepts file paths.

        Args:
            text: Full document text.
            file_name: Used for provenance fields on each ``ChunkRecord``.
            mime_type: MIME type recorded on each chunk.
            collection: Override the instance-level collection for this call.

        Returns:
            List of ``ChunkRecord.to_dict()`` dicts, one per chunk.
        """
        return self._pipeline.ingest_text(text, file_name=file_name, mime_type=mime_type, collection=collection)

    def _resolve_ingest_path(self, path: str) -> str:
        """Confine a caller-supplied ingest *path* to ``documents_root`` (#126).

        When no root is configured the path is returned unchanged (historical
        behaviour). Otherwise the path is treated as relative to the root and each
        component is walked through blue-data's ``contain()``, which rejects
        absolute paths, ``..`` traversal, and symlinks that escape the root."""
        root = self._documents_root
        if not root:
            return path
        parts = Path(path).parts
        if not parts:
            raise ValueError(f"DOCUMENT.INGEST path is empty: {path!r}")
        current = str(Path(root).resolve())
        for part in parts:
            current = contain(current, part)
        return current

    def _ingest(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        path = request.get("path")
        if not path:
            raise ValueError("DOCUMENT.INGEST requires a 'path' field")

        path = self._resolve_ingest_path(path)
        suffix = Path(path).suffix.lower()
        self.properties["connection"]["protocol"] = self._PROTOCOL_BY_SUFFIX.get(suffix, "document")

        collection = collection_override or request.get("collection")
        return self._pipeline.ingest(path, collection=collection)

    def _search(self, request: Dict[str, Any], collection_override=None):
        """Implement DOCUMENT.SEARCH.

        Optional request fields:
            ``k`` — number of results (default 5)
            ``filter`` — pre-filter dict applied before KNN scoring
            ``mode`` — ``"vector"`` (default), ``"hybrid"``, ``"bm25"``, or ``"rerank"``
            ``rich`` — when ``true``, returns a structured dict with provenance,
                neighbor context, and ``by_document`` grouping instead of a flat list
            ``context_chunks`` — neighbor chunks per result when ``rich=true`` (default 1)
        """
        query_text = request.get("query")
        if not query_text:
            raise ValueError("DOCUMENT.SEARCH requires a 'query' field")

        collection = collection_override or request.get("collection")
        k = request.get("k", 5)
        filters = request.get("filter")
        mode = request.get("mode", "vector")
        rich = request.get("rich", False)
        context_chunks = request.get("context_chunks", 1)
        return self._pipeline.search(
            query_text, k=k, collection=collection, filters=filters,
            mode=mode, rich=rich, context_chunks=context_chunks,
        )

    def _ensure_index(self, request: Dict[str, Any], collection_override=None) -> None:
        collection = collection_override or request.get("collection")
        self._pipeline.ensure_index(collection=collection)

    def execute_query_nl(self, query: str, database=None, collection=None, schema=None, llm_fn=None, optional_properties=None, allow_write_ops: bool = False):
        """Parse a natural language instruction and execute the corresponding op.

        Uses the ``NLQueryParser`` passed as ``nl_parser`` at construction, or
        lazily builds a default one when none was provided — so a registry-hosted
        source rebuilt from properties can still be driven by NL. The ``schema`` and
        ``llm_fn`` parameters are accepted to satisfy the ``DataSource`` ABC contract
        but are not used — doc-source uses its own ``NLQueryParser`` for intent
        extraction.

        **Security:** natural-language input is untrusted — it may be an end-user
        question or text an agent forwards from a retrieved document. By default
        only read-only ops (``SEARCH``, ``FIND``, ``FIND_THEN_SEARCH``,
        ``GET_RECORD``, ``GRAPH_*``) are permitted, so a prompt-injected
        instruction cannot ingest an arbitrary path or delete documents. Set
        ``allow_write_ops=True`` to permit state-changing ops (``INGEST``,
        ``DELETE``, ``ENSURE_INDEX``) — only for trusted callers.

        Args:
            query: Natural language instruction, e.g.
                ``"find the termination clause in the contracts collection"``.
            allow_write_ops: When ``True``, permit state-changing ops. Defaults
                to ``False`` (read-only).

        Returns:
            The result of the dispatched ``execute_query`` call.

        Raises:
            ValueError: If no ``nl_parser`` was provided, if the parser returns an
                invalid op dict, or if the parsed op is state-changing and
                ``allow_write_ops`` is ``False``.
        """
        if self._nl_parser is None:
            # Lazily build the default parser so a registry-hosted source — rebuilt
            # from (name, properties) with no nl_parser injected — can still answer
            # NL queries rather than raising (DataSource contract, #132). schema and
            # llm_fn are accepted for the ABC signature but not used: doc-source uses
            # its own NLQueryParser for intent extraction.
            self._nl_parser = NLQueryParser()
        op_dict = self._nl_parser.parse(query)
        if not isinstance(op_dict, dict) or not isinstance(op_dict.get("op"), str):
            raise ValueError(f"NLQueryParser returned an invalid op dict: {op_dict!r}")
        op = op_dict["op"]
        if not allow_write_ops and op not in _READ_ONLY_NL_OPS:
            raise ValueError(
                f"Natural-language queries are restricted to read-only ops by default; "
                f"{op!r} is not permitted. Pass allow_write_ops=True to allow "
                f"state-changing ops (DOCUMENT.INGEST, DOCUMENT.DELETE, DOCUMENT.ENSURE_INDEX)."
            )
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

    def _find(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement DOCUMENT.FIND — filter schema records by extracted field values.

        Filter syntax (all conditions ANDed):
            ``document_type``           exact match, case-insensitive
            ``fields.<name>``           exact match or operator dict
            ``{"contains": "..."}``     substring match
            ``{"gt": N}``               numeric or lexicographic comparison
            ``{"lt": N}``, ``{"gte": N}``, ``{"lte": N}``

        Returns a list of schema records — one per matching document (or a
        :class:`FindResult` when ``coverage=True`` is passed in the request).
        Each record includes ``document_id``, ``document_type``, ``confidence``,
        ``classifier``, and ``fields``.
        """
        return self._pipeline.find_records(
            request.get("filter"),
            coverage=bool(request.get("coverage", False)),
        )

    def _find_then_search(self, request: Dict[str, Any], collection_override=None) -> List[Dict[str, Any]]:
        """Implement DOCUMENT.FIND_THEN_SEARCH.

        Filters DocumentRecords by ``find_filter``, then runs a text search
        scoped to each matched document. Results from all matched documents are
        merged and returned sorted by score, capped at ``k``.

        Required request fields:
            ``query`` — text to search for within matched documents

        Optional request fields:
            ``find_filter`` — same filter dict accepted by DOCUMENT.FIND
            ``k`` — number of results (default 5)
            ``collection`` — collection to search in
            ``mode`` — search mode passed to ``pipeline.search`` (default ``"vector"``)
        """
        query_text = request.get("query")
        if not query_text:
            raise ValueError("DOCUMENT.FIND_THEN_SEARCH requires a 'query' field")

        find_filter = request.get("find_filter")
        collection = collection_override or request.get("collection")
        k = request.get("k", 5)
        mode = request.get("mode", "vector")

        # Delegate the merge + mode-correct ranking to the pipeline so the library,
        # REST and CLI share one implementation. Previously three copies diverged and
        # this one sorted vector-mode distances descending → worst match first (#74).
        return self._pipeline.find_then_search(
            query_text,
            find_filter=find_filter,
            k=k,
            collection=collection,
            mode=mode,
        )

    def _get_record(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        document_id = request.get("document_id")
        if not document_id:
            raise ValueError("DOCUMENT.GET_RECORD requires a 'document_id' field")
        schema = self._pipeline.get_record(document_id)
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

    def _graph_neighbors(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement DOCUMENT.GRAPH_NEIGHBORS.

        Required request fields:
            ``entity`` — entity name to search for

        Optional request fields:
            ``entity_type`` — restrict to one type (PROPER_NOUN, DATE, …)
            ``k`` — maximum results (default 10)
        """
        entity = request.get("entity")
        if not entity:
            raise ValueError("DOCUMENT.GRAPH_NEIGHBORS requires an 'entity' field")
        return self._pipeline.graph_neighbors(
            entity_name=entity,
            entity_type=request.get("entity_type"),
            k=request.get("k", 10),
        )

    def _graph_query(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement DOCUMENT.GRAPH_QUERY — general graph traversal.

        Required request fields:
            ``node_id`` — graph node to start from
                (e.g. ``"doc:doc_abc123"``, ``"entity:PROPER_NOUN:acme corp"``)

        Optional request fields:
            ``edge_type`` — edge type to follow (e.g. ``"HAS_CHUNK"``, ``"NEXT"``,
                ``"APPEARS_IN"``). ``None`` / omitted traverses all edge types.
            ``direction`` — ``"out"``, ``"in"``, or ``"both"`` (default)
            ``k`` — maximum number of results (default: all)

        Returns:
            List of dicts with ``node_id``, ``label``, ``properties``,
            ``edge_type``, and ``direction``.
        """
        node_id = request.get("node_id")
        if not node_id:
            raise ValueError("DOCUMENT.GRAPH_QUERY requires a 'node_id' field")
        return self._pipeline.graph_query(
            node_id=node_id,
            edge_type=request.get("edge_type"),
            direction=request.get("direction", "both"),
            k=request.get("k"),
        )
