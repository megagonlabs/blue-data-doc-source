import re
import threading
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from blue_data.core.utils.similarity_utils import compute_bm25_score, normalize_bm25_scores, compute_vector_score
from blue_data.source.vector.vector_source import VectorSource

from .chunkers import Chunker, SectionChunker
from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .chunk_record import ChunkRecord
from .graph import GraphIndexer
from .parsers import DocxParser, Parser, PDFParser, TextParser
from .find_result import CoverageReport, FindResult
from .schema.base import DocumentClassifier, FieldExtractor
from .schema.schema_store import MemoryRecordStore, RecordStore, _get_nested

if TYPE_CHECKING:
    from blue_data.source.graph.graph_source import GraphSource


class ColdChunkIndexWarning(UserWarning):
    """Warned when ``bm25``/``rerank`` search or ``rich`` neighbor context is
    requested but the in-process chunk index has no entries for the target
    collection.

    ``_chunk_index`` / ``_chunk_embeddings`` / ``_chunk_collection`` are built
    during :meth:`DocumentPipeline.ingest` and live only in the ingesting
    process; they are **not** rebuilt from the vector store after a restart or
    in a separate process. BM25 and rerank score against this index and
    ``rich`` neighbor context reads from it, so when it is empty those features
    silently degrade (empty / wrong results, omitted context). This warning
    makes that failure loud rather than silent (#50). ``vector`` search is
    unaffected — it queries the persistent vector store directly; ``hybrid``
    falls back to vector-only KNN and warns about the fallback.

    Filter or escalate it like any warning category, e.g.::

        warnings.simplefilter("error", ColdChunkIndexWarning)
    """


# Chars that need escaping inside Redis Search TAG filter values.
_TAG_SPECIAL_RE = re.compile(r'([,.<>{}\[\]"\'`:;!@#$%^&*\-+=~|/ ])')

# Fields indexed as TAG in the vector index — used for pre-filtering.
_FILTER_TAG_FIELDS = ("document_id", "document_type", "file_name", "section_title", "mime_type", "content_type")

# All filter keys accepted by search(): the TAG fields plus the numeric page range.
_ALLOWED_FILTER_KEYS = set(_FILTER_TAG_FIELDS) | {"page_start"}

# Extra fields passed to create_index so Redis Search can (a) filter on them and
# (b) project chunk content back on search — a Redis FT.SEARCH RETURN can only
# return *indexed* fields, so chunk_id/text must be indexed for search() to carry
# content rather than just {id, score} (#71).
_INDEX_EXTRA_FIELDS = [
    {"path": "$.chunk_id",      "name": "chunk_id",      "field_type": "TAG"},
    {"path": "$.document_id",   "name": "document_id",   "field_type": "TAG"},
    {"path": "$.document_type", "name": "document_type", "field_type": "TAG"},
    {"path": "$.file_name",     "name": "file_name",     "field_type": "TAG"},
    {"path": "$.section_title", "name": "section_title", "field_type": "TAG"},
    {"path": "$.mime_type",     "name": "mime_type",     "field_type": "TAG"},
    {"path": "$.page_start",    "name": "page_start",    "field_type": "NUMERIC"},
    {"path": "$.page_end",      "name": "page_end",      "field_type": "NUMERIC"},
    {"path": "$.content_type",  "name": "content_type",  "field_type": "TAG"},
    {"path": "$.text",          "name": "text",          "field_type": "TEXT"},
]

# Fields projected back from the vector store on every search so hits carry chunk
# content + provenance, not just {id, score} (#71). Each must be indexed above.
_SEARCH_RETURN_FIELDS = [
    "chunk_id", "text", "document_id", "document_type", "file_name",
    "section_title", "page_start", "page_end", "mime_type", "content_type",
]

# Bounds on caller-supplied values. Unbounded k / context_chunks / chunk_count let
# one request drive linear CPU (the rich-context loop) or backend round-trips (the
# delete loop) — e.g. chunk_count=10**9 is ~a billion delete calls. Reject
# out-of-range values rather than silently degrading or pinning a worker (#80).
_MAX_SEARCH_K = 100
_MAX_CONTEXT_CHUNKS = 5
_MAX_DELETE_CHUNK_COUNT = 100_000


def _validate_bounded_int(name: str, value: Any, lo: int, hi: int) -> None:
    """Require *value* to be an int in ``[lo, hi]``; raise ``ValueError`` otherwise.

    ``bool`` is rejected explicitly (it is an ``int`` subclass) so ``True``/``False``
    can't slip through as 1/0.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")


def _validate_filters(filters: Optional[Dict[str, Any]]) -> None:
    """Reject unknown filter keys and ill-typed values instead of silently dropping them.

    Raises:
        ValueError: on an unknown key, a non-string TAG value, or a ``page_start``
            that is neither an int nor a ``{"gte": int, "lte": int}`` range.
    """
    if not filters:
        return
    unknown = set(filters) - _ALLOWED_FILTER_KEYS
    if unknown:
        raise ValueError(
            f"Unknown search filter key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(_ALLOWED_FILTER_KEYS)}"
        )
    for field in _FILTER_TAG_FIELDS:
        if field in filters and not isinstance(filters[field], str):
            raise ValueError(
                f"Filter {field!r} must be a string, got {type(filters[field]).__name__}"
            )
    page = filters.get("page_start")
    if page is not None:
        if isinstance(page, bool) or not isinstance(page, (int, dict)):
            raise ValueError(
                "Filter 'page_start' must be an int or a {'gte': int, 'lte': int} range, "
                f"got {type(page).__name__}"
            )
        if isinstance(page, dict):
            bad = set(page) - {"gte", "lte"}
            if bad:
                raise ValueError(
                    f"'page_start' range supports only 'gte'/'lte', got {sorted(bad)}"
                )
            for bound in ("gte", "lte"):
                if bound in page and (isinstance(page[bound], bool) or not isinstance(page[bound], int)):
                    raise ValueError(f"'page_start.{bound}' must be an int")


def _escape_redis_tag(value: str) -> str:
    """Escape special chars in a Redis Search TAG filter value."""
    return _TAG_SPECIAL_RE.sub(r'\\\1', value)


def _build_filter_expr(filters: Optional[Dict[str, Any]]) -> str:
    """Translate a filters dict into a Redis Search filter expression.

    Supported keys:
        document_id, document_type, file_name, section_title, mime_type,
        content_type  — exact TAG match
        page_start  — int for exact page, or {"gte": N, "lte": M} for range

    Returns ``"*"`` when filters is None or empty.
    """
    if not filters:
        return "*"

    parts: List[str] = []

    for field in _FILTER_TAG_FIELDS:
        value = filters.get(field)
        if value is not None:
            parts.append(f"@{field}:{{{_escape_redis_tag(str(value))}}}")

    page = filters.get("page_start")
    if page is not None:
        if isinstance(page, int):
            parts.append(f"@page_start:[{page} {page}]")
        elif isinstance(page, dict):
            lo = page.get("gte", "-inf")
            hi = page.get("lte", "+inf")
            parts.append(f"@page_start:[{lo} {hi}]")

    if not parts:
        return "*"
    if len(parts) == 1:
        return parts[0]
    return " ".join(f"({p})" for p in parts)


def _normalize_knn_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise raw vector-store hits so each carries a ``chunk_id`` and float ``score``.

    blue-data's RedisVectorSource returns the Redis key as ``id`` and the score as a
    string; projected fields (incl. ``chunk_id``) are present only when requested via
    ``return_fields``. Derive ``chunk_id`` from the key when absent and coerce ``score``
    to float so downstream dedup (FIND_THEN_SEARCH) and ranking work (#71). In-memory
    backends already return full dicts with a numeric score, so this is a no-op there.
    """
    for hit in hits:
        if not hit.get("chunk_id"):
            key = hit.get("id")
            if isinstance(key, str) and key:
                hit["chunk_id"] = key.rsplit(":", 1)[-1]
        score = hit.get("score")
        if isinstance(score, str):
            try:
                hit["score"] = float(score)
            except ValueError:
                pass
    return hits


class DocumentPipeline:
    """Orchestrates the full document ingestion pipeline.

    Chains parsing → chunking → enrichment → embedding → ChunkRecord
    construction → optional vector store upsert. Auto-detects ``.txt`` vs
    ``.pdf`` by file extension.

    All optional components (``embedding_model``, ``vector_source``,
    ``enricher``) can be omitted. Upsert is skipped unless both
    ``embedding_model`` and ``vector_source`` are set.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        chunker: Optional[Chunker] = None,
        parser: Optional[Parser] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_source: Optional[VectorSource] = None,
        enricher: Optional[Enricher] = None,
        collection: str = "documents",
        ocr_fallback: bool = False,
        classifier: Optional[DocumentClassifier] = None,
        field_extractor: Optional[FieldExtractor] = None,
        record_store: Optional[RecordStore] = None,
        graph_source: Optional["GraphSource"] = None,
        graph_database: str = "documents",
    ):
        """
        Args:
            chunk_size: Maximum words per chunk. Passed to the default SectionChunker.
            overlap: Word overlap between consecutive chunks.
            chunker: Chunker instance to use. Defaults to ``SectionChunker``, which
                detects section headers and falls back to window chunking.
            parser: Custom ``Parser`` instance. When provided, ``ingest()`` calls
                ``parser.parse(path)`` instead of the built-in ``TextParser`` /
                ``PDFParser``. Use this to plug in LlamaIndex, a vision model, or
                any other external document processor.
            embedding_model: Embedding model to use. When ``None``, no
                embeddings are generated and upsert is skipped.
            vector_source: VectorSource backend from blue-data. When ``None``,
                chunks are returned but not persisted.
            enricher: Enrichment step. When ``None``, ``entities`` and
                ``semantic_hints`` are left empty on each record.
            collection: Default collection name used when upserting documents.
            ocr_fallback: When ``True``, passes ``ocr_fallback=True`` to the
                built-in ``PDFParser`` so scanned PDF pages are OCR'd via
                Tesseract. Has no effect when a custom ``parser=`` is provided.
                Requires the ``[ocr]`` extras.
            graph_source: Optional blue-data ``GraphSource`` backend
                (``MemoryGraphSource``, ``FileGraphSource``, or
                ``NEO4JSource``). When set, every ingested document is indexed
                into the graph with ``Document``, ``Chunk``, and ``Entity``
                nodes. Enables ``graph_neighbors()`` retrieval. When ``None``
                (default) the graph layer is disabled and no existing behaviour
                changes.
            graph_database: Graph database name used by ``graph_source``.
                Defaults to ``"documents"``.
        """
        self.text_parser = TextParser()
        self.pdf_parser = PDFParser(ocr_fallback=ocr_fallback)
        self.docx_parser = DocxParser()
        self.parser = parser
        self.chunker = chunker or SectionChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = embedding_model
        self.vector_source = vector_source
        self.enricher = enricher
        self.classifier = classifier
        self.field_extractor = field_extractor
        self.record_store: RecordStore = record_store or MemoryRecordStore()
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap
        # path → (document_id, chunk_count, collection) — used to delete stale chunks
        self._doc_registry: Dict[str, Tuple[str, int, str]] = {}
        # collections for which create_index has already succeeded this session
        self._index_created: Set[str] = set()
        # Guards the in-process chunk indexes below so a concurrent BM25/hybrid/rerank
        # search iterating them cannot race an ingest/delete mutation (#87:
        # "dictionary changed size during iteration").
        self._index_lock = threading.Lock()
        # chunk_id → chunk dict — populated during ingest; used for neighbor context and BM25
        self._chunk_index: Dict[str, Dict[str, Any]] = {}
        # chunk_id → collection name — needed to scope BM25 to the correct collection
        self._chunk_collection: Dict[str, str] = {}
        # chunk_id → embedding vector — used by rerank mode to avoid re-embedding candidates
        self._chunk_embeddings: Dict[str, List[float]] = {}
        # optional graph indexer — None when graph_source is not configured
        self._graph_indexer: Optional[GraphIndexer] = None
        if graph_source is not None:
            self._graph_indexer = GraphIndexer(graph_source, database=graph_database)
            self._graph_indexer.ensure_database()

    def ingest(self, path: str, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse, chunk, enrich, and embed a file.

        Args:
            path: Path to a ``.txt`` or ``.pdf`` file.
            collection: Override the instance-level collection for this call only.

        Returns:
            List of ``ChunkRecord.to_dict()`` dicts, one per chunk. Each dict
            includes provenance fields (``document_id``, ``chunk_id``,
            ``page_start``/``page_end`` for PDFs) and optional ``embedding``.

        Raises:
            ValueError: If the file extension is not ``.txt`` or ``.pdf``.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        effective_collection = collection or self.collection

        # Custom parser path — e.g. PlumberPDFParser for table extraction,
        # a vision model parser for scanned images, or Docling for rich layouts.
        if self.parser is not None:
            blocks = self.parser.parse_blocks(path)
            mime_type = "application/pdf" if suffix == ".pdf" else "text/plain"
            document_id = self._file_document_id(file_path)
            self._cleanup_stale_chunks(file_path, document_id)
            records = self._ingest_from_blocks(
                blocks=blocks,
                file_name=file_path.name,
                source_path=str(file_path),
                mime_type=mime_type,
                document_id=document_id,
                collection=effective_collection,
            )
            self._register_doc(file_path, document_id, len(records), effective_collection)
            return records

        if suffix == ".txt":
            return self._ingest_txt(file_path, effective_collection)
        elif suffix == ".pdf":
            return self._ingest_pdf(file_path, effective_collection)
        elif suffix == ".docx":
            return self._ingest_docx(file_path, effective_collection)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _make_record(
        self,
        document_id: str,
        chunk_id: str,
        file_name: str,
        source_path: Optional[str],
        chunk_text: str,
        chunk_index: int,
        ingested_at: str,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        section_title: Optional[str] = None,
        mime_type: str = "text/plain",
        page_ocr_meta: Optional[Dict[str, Any]] = None,
        document_type: Optional[str] = None,
        content_type: Optional[str] = None,
        table_data: Optional[Dict[str, Any]] = None,
    ) -> ChunkRecord:
        embedding = None
        if self.embedding_model is not None:
            embedding = self.embedding_model.embed_text(chunk_text)
            if len(embedding) != self.embedding_model.dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_model.dim}, "
                    f"got {len(embedding)}"
                )

        entities = []
        semantic_hints: Dict[str, Any] = {}
        if self.enricher is not None:
            enriched = self.enricher.enrich(chunk_text)
            entities = enriched.get("entities", [])
            semantic_hints = enriched.get("semantic_hints", {})

        metadata: Dict[str, Any] = {
            "source_path": source_path,
            "chunk_index": chunk_index,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "ingested_at": ingested_at,
            "embedding_model": (
                self.embedding_model.__class__.__name__
                if self.embedding_model is not None
                else None
            ),
        }
        if page_ocr_meta:
            metadata.update(page_ocr_meta)

        return ChunkRecord(
            document_id=document_id,
            chunk_id=chunk_id,
            file_name=file_name,
            mime_type=mime_type,
            text=chunk_text,
            page_start=page_start,
            page_end=page_end,
            section_title=section_title,
            entities=entities,
            semantic_hints=semantic_hints,
            embedding=embedding,
            metadata=metadata,
            document_type=document_type,
            content_type=content_type,
            table=table_data,
        )

    @staticmethod
    def _file_document_id(file_path: Path) -> str:
        digest = sha256(file_path.read_bytes()).hexdigest()[:16]
        return f"doc_{digest}"

    def _upsert(self, chunk_id: str, record_dict: Dict[str, Any], collection: str) -> None:
        if self.vector_source is not None and record_dict.get("embedding") is not None:
            self.vector_source.upsert(
                collection=collection,
                doc_id=chunk_id,
                doc=record_dict,
            )

    def ensure_index(self, collection: Optional[str] = None) -> None:
        """Create the vector index for a collection if it does not already exist.

        Idempotent: safe to call multiple times. Skips the call if this pipeline
        instance already created the index this session. If the backend reports
        that the index already exists, the error is suppressed.

        Args:
            collection: Collection / index name. Defaults to the instance-level collection.
        """
        if self.vector_source is None or self.embedding_model is None:
            return
        target = collection or self.collection
        if target in self._index_created:
            return
        try:
            self.vector_source.create_index(
                index_name=target,
                collection=target,
                dim=self.embedding_model.dim,
                extra_fields=_INDEX_EXTRA_FIELDS,
            )
        except TypeError:
            # Vector source doesn't accept extra_fields (e.g. MemoryVectorSource).
            try:
                self.vector_source.create_index(
                    index_name=target,
                    collection=target,
                    dim=self.embedding_model.dim,
                )
            except Exception as exc2:
                if "already exists" not in str(exc2).lower():
                    raise
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._index_created.add(target)

    def _cleanup_stale_chunks(self, file_path: Path, new_doc_id: str) -> None:
        """Delete chunks from the previous version of a file when content has changed.

        Best-effort: individual delete failures are silently ignored so a partially
        cleaned-up state does not block the new ingest.

        Note: the registry is in-process only. Stale chunks from previous process
        runs are not cleaned up automatically; use DOCUMENT.DELETE for that.
        """
        key = str(file_path)
        if key not in self._doc_registry:
            return
        old_doc_id, old_count, old_collection = self._doc_registry[key]
        if old_doc_id == new_doc_id:
            return  # same content — upsert will overwrite in place
        # Drop the stale structured record for the previous content-hash ID (#62),
        # before the vector_source guard so it happens even with no vector store.
        self.record_store.delete(old_doc_id)
        if self._graph_indexer is not None:
            self._graph_indexer.delete_document(old_doc_id)
        if self.vector_source is None:
            return
        for idx in range(old_count):
            chunk_id = f"{old_doc_id}_chunk_{idx:06d}"
            with self._index_lock:
                self._chunk_index.pop(chunk_id, None)
                self._chunk_embeddings.pop(chunk_id, None)
            try:
                self.vector_source.delete(collection=old_collection, doc_id=chunk_id)
            except Exception:
                pass

    def _register_doc(self, file_path: Path, doc_id: str, chunk_count: int, collection: str) -> None:
        self._doc_registry[str(file_path)] = (doc_id, chunk_count, collection)

    def ingest_text(
        self,
        text: str,
        file_name: str,
        mime_type: str = "text/plain",
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk, enrich, and embed pre-extracted text, bypassing file parsing.

        Use this when text has already been extracted by an external tool
        (e.g. LlamaIndex, a vision model, or a custom extractor) and you want
        to feed it directly into the chunking → embedding → storage pipeline.

        Args:
            text: Full document text.
            file_name: Used for provenance fields on each ``ChunkRecord``.
            mime_type: MIME type recorded on each chunk.
            collection: Override the instance-level collection for this call.

        Returns:
            List of ``ChunkRecord.to_dict()`` dicts, one per chunk.
        """
        document_id = "doc_" + sha256(text.encode()).hexdigest()[:16]
        return self._ingest_from_pages(
            pages=[{"text": text}],
            file_name=file_name,
            source_path=None,
            mime_type=mime_type,
            document_id=document_id,
            collection=collection or self.collection,
        )

    def ingest_pages(
        self,
        pages: List[Dict[str, Any]],
        file_name: str,
        mime_type: str = "application/pdf",
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk, enrich, and embed pre-extracted pages, bypassing file parsing.

        Use this when page-level text has already been extracted by an external
        tool (e.g. a vision model or LlamaIndex's PDF reader) and you want to
        feed it directly into the chunking → embedding → storage pipeline.

        Args:
            pages: List of dicts with ``"text": str`` and optional
                ``"page_num": int`` (1-based). ``page_num`` is propagated to
                ``page_start`` / ``page_end`` on each ``ChunkRecord``.
            file_name: Used for provenance fields on each ``ChunkRecord``.
            mime_type: MIME type recorded on each chunk.
            collection: Override the instance-level collection for this call.

        Returns:
            List of ``ChunkRecord.to_dict()`` dicts, one per chunk.
        """
        combined = "".join(p.get("text", "") for p in pages)
        document_id = "doc_" + sha256(combined.encode()).hexdigest()[:16]
        return self._ingest_from_pages(
            pages=pages,
            file_name=file_name,
            source_path=None,
            mime_type=mime_type,
            document_id=document_id,
            collection=collection or self.collection,
        )

    def _ingest_from_pages(
        self,
        pages: List[Dict[str, Any]],
        file_name: str,
        source_path: Optional[str],
        mime_type: str,
        document_id: str,
        collection: str,
    ) -> List[Dict[str, Any]]:
        """Classify, chunk, enrich, embed, and upsert a list of page dicts.

        Runs classification and field extraction once across the full document
        text before chunking — so document_type propagates to every chunk
        without repeated LLM calls. Each page is then chunked independently;
        the chunk_index counter is global across all pages so chunk IDs are
        continuous (not restarting per page). OCR metadata (``ocr``,
        ``ocr_confidence``) from the parser is forwarded to each chunk's
        metadata. Graph indexing runs once after all chunks are produced.

        Used by the built-in parsers (TextParser, PDFParser, DocxParser) and
        by ingest_text() / ingest_pages(). For block-aware parsers that return
        typed content (e.g. PlumberPDFParser), use _ingest_from_blocks instead.
        """
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []
        chunk_index = 0

        # Run document classification / schema extraction once per document. The
        # DocumentRecord is built here but written only after every chunk upsert
        # succeeds (below) — so a crash / vector-store error mid-loop leaves no
        # DocumentRecord to find, and re-ingesting from the source is a clean upsert (#49).
        document_type: Optional[str] = None
        schema_record: Optional[Dict[str, Any]] = None
        if self.classifier is not None or self.field_extractor is not None:
            full_text = " ".join(p.get("text", "") for p in pages)
            if self.classifier is not None:
                document_type, confidence = self.classifier.classify(full_text)
            else:
                document_type, confidence = "unknown", 0.0
            extracted_fields: Dict[str, Any] = {}
            if self.field_extractor is not None:
                extracted_fields = self.field_extractor.extract(full_text, document_type)
            classifier_name = (
                "llm" if self.field_extractor is not None else "keyword"
            )
            schema_record = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_schema",
                "document_type": document_type,
                "fields": extracted_fields,
                "classifier": classifier_name,
                "confidence": confidence,
            }

        for page in pages:
            page_num = page.get("page_num")
            page_ocr_meta: Dict[str, Any] = {}
            if page.get("ocr"):
                page_ocr_meta["ocr"] = True
            if page.get("ocr_confidence") is not None:
                page_ocr_meta["ocr_confidence"] = page["ocr_confidence"]

            chunks = self.chunker.chunk_with_titles(page["text"])
            for section_title, chunk_text in chunks:
                chunk_id = f"{document_id}_chunk_{chunk_index:06d}"
                record = self._make_record(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    file_name=file_name,
                    source_path=source_path,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    ingested_at=ingested_at,
                    page_start=page_num,
                    page_end=page_num,
                    section_title=section_title,
                    mime_type=mime_type,
                    page_ocr_meta=page_ocr_meta or None,
                    document_type=document_type,
                )
                record_dict = record.to_dict()
                records.append(record_dict)
                with self._index_lock:
                    self._chunk_index[chunk_id] = record_dict
                    self._chunk_collection[chunk_id] = collection
                    if record.embedding is not None:
                        self._chunk_embeddings[chunk_id] = record.embedding
                self._upsert(chunk_id, record_dict, collection)
                chunk_index += 1

        # Every chunk upserted — now persist the DocumentRecord, stamped with the final
        # chunk_count so a reader can validate the record against the vector store (#49).
        if schema_record is not None:
            schema_record["chunk_count"] = chunk_index
            self.record_store.upsert(schema_record)

        if self._graph_indexer is not None:
            self._graph_indexer.index_document(document_id, file_name, document_type, records)

        return records

    def _ingest_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        file_name: str,
        source_path: Optional[str],
        mime_type: str,
        document_id: str,
        collection: str,
    ) -> List[Dict[str, Any]]:
        """Ingest typed content blocks, keeping table blocks atomic.

        Text blocks are chunked normally by the configured chunker. Table
        blocks bypass the chunker entirely — each table becomes exactly one
        chunk with ``content_type="table"`` and a structured ``table`` field.
        """
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []
        chunk_index = 0

        # DocumentRecord built here, written only after every chunk upsert succeeds (#49).
        document_type: Optional[str] = None
        schema_record: Optional[Dict[str, Any]] = None
        if self.classifier is not None or self.field_extractor is not None:
            full_text = " ".join(b.get("text", "") for b in blocks)
            if self.classifier is not None:
                document_type, confidence = self.classifier.classify(full_text)
            else:
                document_type, confidence = "unknown", 0.0
            extracted_fields: Dict[str, Any] = {}
            if self.field_extractor is not None:
                extracted_fields = self.field_extractor.extract(full_text, document_type)
            classifier_name = "llm" if self.field_extractor is not None else "keyword"
            schema_record = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_schema",
                "document_type": document_type,
                "fields": extracted_fields,
                "classifier": classifier_name,
                "confidence": confidence,
            }

        for block in blocks:
            block_type = block.get("type", "text")
            page_num = block.get("page_num")

            if block_type == "table":
                chunk_id = f"{document_id}_chunk_{chunk_index:06d}"
                record = self._make_record(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    file_name=file_name,
                    source_path=source_path,
                    chunk_text=block.get("text", ""),
                    chunk_index=chunk_index,
                    ingested_at=ingested_at,
                    page_start=page_num,
                    page_end=page_num,
                    mime_type=mime_type,
                    document_type=document_type,
                    content_type="table",
                    table_data=block.get("table"),
                )
                record_dict = record.to_dict()
                records.append(record_dict)
                with self._index_lock:
                    self._chunk_index[chunk_id] = record_dict
                    self._chunk_collection[chunk_id] = collection
                    if record.embedding is not None:
                        self._chunk_embeddings[chunk_id] = record.embedding
                self._upsert(chunk_id, record_dict, collection)
                chunk_index += 1
            else:
                chunks = self.chunker.chunk_with_titles(block.get("text", ""))
                for section_title, chunk_text in chunks:
                    chunk_id = f"{document_id}_chunk_{chunk_index:06d}"
                    record = self._make_record(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        file_name=file_name,
                        source_path=source_path,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        ingested_at=ingested_at,
                        page_start=page_num,
                        page_end=page_num,
                        section_title=section_title,
                        mime_type=mime_type,
                        document_type=document_type,
                        content_type="text",
                    )
                    record_dict = record.to_dict()
                    records.append(record_dict)
                    with self._index_lock:
                        self._chunk_index[chunk_id] = record_dict
                        self._chunk_collection[chunk_id] = collection
                        if record.embedding is not None:
                            self._chunk_embeddings[chunk_id] = record.embedding
                    self._upsert(chunk_id, record_dict, collection)
                    chunk_index += 1

        # Every chunk upserted — now persist the DocumentRecord with the final chunk_count (#49).
        if schema_record is not None:
            schema_record["chunk_count"] = chunk_index
            self.record_store.upsert(schema_record)

        if self._graph_indexer is not None:
            self._graph_indexer.index_document(document_id, file_name, document_type, records)

        return records

    def get_record(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return the DocumentRecord for a document, or ``None`` if not available.

        Records are populated during ingest when a ``classifier`` is set.
        """
        return self.record_store.get(document_id)

    def find_records(
        self,
        filter: Optional[Dict[str, Any]] = None,
        coverage: bool = False,
    ) -> "List[Dict[str, Any]] | FindResult":
        """Return all DocumentRecords matching *filter*.

        See ``RecordStore.find`` for the filter syntax (dot notation, operators).
        Returns every record when *filter* is ``None`` or ``{}``.

        Args:
            filter: Structured filter dict (dot notation + operators). ``None`` returns all.
            coverage: When ``True``, return a :class:`FindResult` instead of a plain list.
                The result carries a :class:`CoverageReport` indicating how many documents
                were missing the queried ``fields.*`` keys — alerting the caller to
                possible false negatives from incomplete extraction.
        """
        if not coverage:
            return self.record_store.find(filter)

        all_records = self.record_store.find(None)
        matched = self.record_store.find(filter)

        field_keys = [k for k in (filter or {}) if k.startswith("fields.")]
        documents_missing_field = (
            sum(
                1
                for r in all_records
                if any(_get_nested(r, k) is None for k in field_keys)
            )
            if field_keys
            else 0
        )

        documents_considered = len(all_records)
        field_coverage_rate = (
            (documents_considered - documents_missing_field) / documents_considered
            if documents_considered > 0
            else 0.0
        )

        if documents_considered == 0:
            status = "no_data"
        elif documents_missing_field == 0:
            status = "ok"
        else:
            status = "partial"

        warning: Optional[str] = None
        if status == "partial":
            names = ", ".join(f"'{k}'" for k in field_keys)
            warning = (
                f"Result may be incomplete: {documents_missing_field} of "
                f"{documents_considered} documents are missing the field(s) {names}."
            )

        return FindResult(
            records=matched,
            status=status,
            coverage=CoverageReport(
                documents_considered=documents_considered,
                documents_matched=len(matched),
                documents_missing_field=documents_missing_field,
                field_coverage_rate=field_coverage_rate,
            ),
            warning=warning,
        )

    def graph_neighbors(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find chunks that mention *entity_name* via graph traversal.

        Requires ``graph_source`` to have been set at construction time.

        Args:
            entity_name: Entity text to search for (case-insensitive).
            entity_type: Optional — restrict to one entity type
                (``"PROPER_NOUN"``, ``"DATE"``, ``"MONEY"``, etc.).
                When ``None``, all known entity types are tried.
            k: Maximum number of chunk results to return.

        Returns:
            List of dicts with ``chunk_id``, ``document_id``,
            ``text_preview``, ``entity``, ``entity_type``, and optionally
            ``section_title``.

        Raises:
            ValueError: If no ``graph_source`` was configured.
        """
        if self._graph_indexer is None:
            raise ValueError(
                "graph_neighbors requires a graph_source. "
                "Pass graph_source= when constructing DocumentPipeline."
            )
        return self._graph_indexer.graph_neighbors(entity_name, entity_type=entity_type, k=k)

    def graph_query(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """General graph traversal from *node_id*.

        Delegates directly to ``GraphSource.get_neighbors`` via
        ``GraphIndexer.get_neighbors``. Use this for arbitrary traversal:
        walking NEXT/PREV chains, inspecting Section nodes, finding all
        documents an entity appears in via APPEARS_IN, etc.

        Args:
            node_id: Graph node ID to start from (e.g. ``doc:doc_abc123``,
                ``chunk:...``, ``entity:PROPER_NOUN:acme corp``).
            edge_type: Edge type to follow. ``None`` traverses all types.
            direction: ``"out"``, ``"in"``, or ``"both"`` (default).
            k: Maximum number of results. ``None`` returns all.

        Returns:
            List of dicts with ``node_id``, ``label``, ``properties``,
            ``edge_type``, and ``direction``.

        Raises:
            ValueError: If no ``graph_source`` was configured.
        """
        if self._graph_indexer is None:
            raise ValueError(
                "graph_query requires a graph_source. "
                "Pass graph_source= when constructing DocumentPipeline."
            )
        return self._graph_indexer.get_neighbors(
            node_id, edge_type=edge_type, direction=direction, k=k
        )

    def _ingest_txt(self, file_path: Path, collection: str) -> List[Dict[str, Any]]:
        text = self.text_parser.parse(str(file_path))
        document_id = self._file_document_id(file_path)
        self._cleanup_stale_chunks(file_path, document_id)
        records = self._ingest_from_pages(
            pages=[{"text": text}],
            file_name=file_path.name,
            source_path=str(file_path),
            mime_type="text/plain",
            document_id=document_id,
            collection=collection,
        )
        self._register_doc(file_path, document_id, len(records), collection)
        return records

    def _ingest_pdf(self, file_path: Path, collection: str) -> List[Dict[str, Any]]:
        pages = self.pdf_parser.parse_pages(str(file_path))
        document_id = self._file_document_id(file_path)
        self._cleanup_stale_chunks(file_path, document_id)
        records = self._ingest_from_pages(
            pages=pages,
            file_name=file_path.name,
            source_path=str(file_path),
            mime_type="application/pdf",
            document_id=document_id,
            collection=collection,
        )
        self._register_doc(file_path, document_id, len(records), collection)
        return records

    def _ingest_docx(self, file_path: Path, collection: str) -> List[Dict[str, Any]]:
        text = self.docx_parser.parse(str(file_path))
        document_id = self._file_document_id(file_path)
        self._cleanup_stale_chunks(file_path, document_id)
        records = self._ingest_from_pages(
            pages=[{"text": text}],
            file_name=file_path.name,
            source_path=str(file_path),
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            document_id=document_id,
            collection=collection,
        )
        self._register_doc(file_path, document_id, len(records), collection)
        return records

    def delete(
        self,
        path: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_count: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> int:
        """Delete all chunks for a document from the vector store.

        Two ways to identify the document:

        - Pass ``path``: looks up the in-process registry populated by ``ingest()``.
          Fails with ``ValueError`` if the path is unknown (e.g. ingested in a
          different process). Removes the registry entry after deletion.
        - Pass both ``document_id`` and ``chunk_count``: reconstructs chunk IDs
          directly, enabling cross-process deletion when the caller stores the
          ingest response.

        Args:
            path: File path used during ingestion.
            document_id: Content-hash document ID from ``ChunkRecord.document_id``.
            chunk_count: Total number of chunks (``len(ingest(...))``).
            collection: Collection to delete from, for a ``document_id`` + ``chunk_count``
                delete (defaults to the instance-level collection). Ignored for a
                ``path`` delete, which always uses the collection recorded at ingest.

        Returns:
            Number of chunks actually removed from the vector store — counted from
            each backend's delete return value, so deleting a document that was never
            ingested reports 0 rather than ``chunk_count`` (#78).

        Raises:
            ValueError: If neither ``path`` nor ``document_id + chunk_count`` is
                provided, if ``path`` is not in the registry, if
                ``vector_source`` is not set, or if ``chunk_count`` is not an int
                in ``[1, 100000]`` or exceeds the count stored for the document
                (#80).
        """
        if self.vector_source is None:
            raise ValueError("A vector_source is required for delete")

        doc_id: str
        count: int
        target: str

        if path is not None:
            key = str(Path(path))
            if key not in self._doc_registry:
                raise ValueError(
                    f"No registry entry for {path!r}. "
                    "Pass document_id and chunk_count to delete cross-process."
                )
            # A path delete uses the collection the document was ingested into (from
            # the registry) — a caller-supplied ``collection`` must not redirect it to
            # a collection the document was never stored in (#78).
            doc_id, count, reg_collection = self._doc_registry[key]
            target = reg_collection
        elif document_id is not None and chunk_count is not None:
            _validate_bounded_int("chunk_count", chunk_count, 1, _MAX_DELETE_CHUNK_COUNT)
            doc_id = document_id
            target = collection or self.collection
            # Derive the real chunk count from the in-process registry when this
            # document was ingested here — a caller must not claim more chunks
            # than were stored and drive extra backend delete calls (#80).
            stored = next(
                (c for (d, c, _) in self._doc_registry.values() if d == doc_id),
                None,
            )
            if stored is not None and chunk_count > stored:
                raise ValueError(
                    f"chunk_count {chunk_count} exceeds the {stored} chunk(s) "
                    f"stored for document {doc_id!r}"
                )
            count = chunk_count
        else:
            raise ValueError(
                "Provide either 'path' or both 'document_id' and 'chunk_count'"
            )

        deleted = 0
        for idx in range(count):
            chunk_id = f"{doc_id}_chunk_{idx:06d}"
            with self._index_lock:
                self._chunk_index.pop(chunk_id, None)
                self._chunk_collection.pop(chunk_id, None)
                self._chunk_embeddings.pop(chunk_id, None)
            try:
                # Count what the backend actually removed (0 or 1), not the call.
                deleted += self.vector_source.delete(collection=target, doc_id=chunk_id)
            except Exception:
                pass

        # Remove the structured record too, so DOCUMENT.FIND can't return a
        # document that has been deleted (#62).
        self.record_store.delete(doc_id)

        if path is not None:
            self._doc_registry.pop(str(Path(path)), None)

        if self._graph_indexer is not None:
            self._graph_indexer.delete_document(doc_id)

        return deleted

    @staticmethod
    def _chunk_matches_filters(
        chunk: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True if a chunk dict satisfies all filter constraints."""
        if not filters:
            return True
        for field in _FILTER_TAG_FIELDS:
            value = filters.get(field)
            if value is not None and chunk.get(field) != value:
                return False
        page = filters.get("page_start")
        if page is not None:
            chunk_page = chunk.get("page_start")
            if isinstance(page, int):
                if chunk_page != page:
                    return False
            elif isinstance(page, dict):
                lo = page.get("gte")
                hi = page.get("lte")
                if chunk_page is None:
                    return False
                if lo is not None and chunk_page < lo:
                    return False
                if hi is not None and chunk_page > hi:
                    return False
        return True

    def _has_indexed_chunks(self, collection: str) -> bool:
        """Whether the in-process chunk index holds any chunk for *collection*.

        ``False`` after a restart / in a separate process (the index is built
        during ingest and never rebuilt from the vector store) — the signal used
        to warn that bm25/rerank/neighbor-context will degrade (#50).
        """
        with self._index_lock:
            return any(c == collection for c in self._chunk_collection.values())

    def _warn_cold_index(self, mode: str, collection: str) -> None:
        """Emit a :class:`ColdChunkIndexWarning` describing the degraded feature."""
        if mode == "hybrid":
            detail = (
                "falling back to vector-only KNN (BM25 ranks cannot be fused)"
            )
        elif mode == "context":
            detail = "neighbor context (context.before/after) will be omitted"
        else:  # bm25 / rerank
            detail = "returning no results" if mode == "bm25" else (
                "returning the BM25 shortlist order without re-ranking"
            )
        warnings.warn(
            f"{mode!r} search on collection {collection!r}: the in-process chunk "
            f"index has no entries for this collection, so {detail}. This index is "
            f"built during ingest() and is not rebuilt from the vector store after "
            f"a restart or in a separate process; re-ingest in this process, or use "
            f"mode='vector' which queries the persistent store directly (#50).",
            ColdChunkIndexWarning,
            stacklevel=3,
        )

    def _bm25_candidates(
        self,
        query: str,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Return normalised BM25 scores for all in-collection chunks.

        Scores all chunks in ``_chunk_index`` that belong to ``collection`` and
        pass ``filters``. Returns ``{chunk_id: normalised_score}`` with scores
        in [0, 1]. Chunks scoring 0 are excluded. Returns an empty dict when
        the in-process index has no entries for the collection (e.g. process
        restart with a persistent vector store).
        """
        # Snapshot under the lock so a concurrent ingest/delete can't mutate the
        # index mid-iteration (#87). The filter runs on the snapshot, not the live dict.
        with self._index_lock:
            index_items = list(self._chunk_index.items())
        candidates = {
            cid: rec
            for cid, rec in index_items
            if self._chunk_collection.get(cid) == collection
            and self._chunk_matches_filters(rec, filters)
        }
        if not candidates:
            return {}
        ids = list(candidates.keys())
        # Strip punctuation so "Termination:" matches query token "termination".
        # Include section_title in document text — section headers contain key terms.
        _punct = re.compile(r"[^\w\s]")
        clean = lambda t: _punct.sub(" ", t)  # noqa: E731
        raw = [
            compute_bm25_score(
                clean(query),
                clean(candidates[cid].get("text", "")),
                schema_text=clean(candidates[cid]["section_title"]) if candidates[cid].get("section_title") else None,
            )
            for cid in ids
        ]
        norm = normalize_bm25_scores(raw)
        return {cid: score for cid, score in zip(ids, norm) if score > 0}

    def find_then_search(
        self,
        query: str,
        find_filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        collection: Optional[str] = None,
        mode: str = "vector",
    ) -> List[Dict[str, Any]]:
        """Filter DocumentRecords by *find_filter*, then search within the matches.

        Finds the documents matching *find_filter*, runs :meth:`search` scoped to each
        matched ``document_id``, merges and de-duplicates chunks by ``chunk_id``, ranks
        them, and returns the top *k*.

        Ranking follows the score semantics of *mode*: in ``"vector"`` mode ``score``
        is a cosine **distance** (lower is better), so results are sorted ascending; in
        ``"bm25"``, ``"hybrid"`` and ``"rerank"`` modes ``score`` is a **similarity**
        (higher is better), so they are sorted descending. This is the single,
        mode-correct merge shared by every surface (library op dispatch, REST, CLI);
        previously each sorted differently and vector mode returned the worst match
        first (#74).
        """
        matched = self.find_records(find_filter)
        if not matched:
            return []

        seen: set = set()
        all_results: List[Dict[str, Any]] = []
        for doc in matched:
            doc_id = doc.get("document_id")
            if not doc_id:
                continue
            for chunk in self.search(
                query, k=k, collection=collection,
                filters={"document_id": doc_id}, mode=mode,
            ):
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    all_results.append(chunk)

        # vector mode: score is cosine distance (lower is better) → ascending.
        # bm25 / hybrid / rerank: score is a similarity (higher is better) → descending.
        lower_is_better = mode == "vector"
        sentinel = float("inf") if lower_is_better else float("-inf")
        all_results.sort(
            key=lambda c: c.get("score") if c.get("score") is not None else sentinel,
            reverse=not lower_is_better,
        )
        return all_results[:k]

    def search(
        self,
        query: str,
        k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "vector",
        rich: bool = False,
        context_chunks: int = 1,
    ) -> Any:
        """Search for chunks matching a query.

        Args:
            query: Natural language query string.
            k: Number of results to return.
            collection: Override the instance-level collection for this call only.
            filters: Optional pre-filter. Supported keys: ``document_id``,
                ``document_type``, ``file_name``, ``section_title``, ``mime_type``,
                ``content_type`` (exact match) and ``page_start`` (int or
                ``{"gte": N, "lte": M}``). Unknown keys or ill-typed values raise
                ``ValueError`` rather than being silently ignored.
            mode: Search strategy — ``"vector"`` (default), ``"hybrid"`` (RRF
                fusion of BM25 + KNN), ``"bm25"`` (keyword-only, no vector store
                required), or ``"rerank"`` (BM25 shortlist re-ranked by cosine).
            rich: When ``False`` (default) returns a flat ``list[dict]`` of
                scored chunk dicts. When ``True`` returns a structured ``dict``
                with ``results``, ``by_document`` grouping, per-chunk provenance
                blocks, and optional neighbor ``context``. Use ``rich=True`` when
                you need citations, source grouping, or explainability.
            context_chunks: Only used when ``rich=True``. Number of adjacent
                chunks to include in ``context.before`` / ``context.after`` for
                each result. Set to ``0`` to omit context entirely.

        Returns:
            ``list[dict]`` when ``rich=False`` — each dict has ``score`` plus
            all ChunkRecord fields. ``dict`` when ``rich=True`` — keys:
            ``query``, ``result_count``, ``document_count``, ``results``
            (ranked list with ``rank``, ``score``, ``chunk_id``, ``text``,
            ``provenance``, optional ``context`` and ``document_type``), and
            ``by_document`` (results grouped by document with ``match_count``
            and optional ``top_sections``).

        Raises:
            ValueError: If ``embedding_model`` or ``vector_source`` is not set
                and ``mode`` is ``"vector"`` or ``"hybrid"``; if ``k`` is not an
                int in ``[1, 100]`` or ``context_chunks`` is not an int in
                ``[0, 5]`` (#80).
        """
        _validate_filters(filters)
        _validate_bounded_int("k", k, 1, _MAX_SEARCH_K)
        _validate_bounded_int("context_chunks", context_chunks, 0, _MAX_CONTEXT_CHUNKS)
        target = collection or self.collection

        # ------------------------------------------------------------------
        # Raw retrieval — collect flat list of scored chunk dicts
        # ------------------------------------------------------------------

        if mode == "bm25":
            if not self._has_indexed_chunks(target):
                self._warn_cold_index("bm25", target)
            bm25_scores = self._bm25_candidates(query, target, filters)
            if not bm25_scores:
                raw: List[Dict[str, Any]] = []
            else:
                ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
                raw = []
                for chunk_id, score in ranked[:k]:
                    chunk = dict(self._chunk_index[chunk_id])
                    chunk["score"] = score
                    raw.append(chunk)

        elif mode == "rerank":
            if not self._has_indexed_chunks(target):
                self._warn_cold_index("rerank", target)
            bm25_scores = self._bm25_candidates(query, target, filters)
            if not bm25_scores:
                raw = []
            else:
                bm25_k = max(k * 4, 20)
                shortlist = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:bm25_k]
                if self.embedding_model is None or not any(
                    cid in self._chunk_embeddings for cid, _ in shortlist
                ):
                    raw = [
                        {**dict(self._chunk_index[cid]), "score": score}
                        for cid, score in shortlist[:k]
                    ]
                else:
                    query_vector = self.embedding_model.embed_text(query)
                    reranked: List[Tuple[float, str]] = []
                    for chunk_id, _ in shortlist:
                        emb = self._chunk_embeddings.get(chunk_id)
                        if emb is None:
                            continue
                        reranked.append((compute_vector_score(query_vector, emb), chunk_id))
                    reranked.sort(key=lambda x: x[0], reverse=True)
                    raw = [
                        {**dict(self._chunk_index[cid]), "score": score}
                        for score, cid in reranked[:k]
                    ]

        else:
            # vector or hybrid — both require embedding_model and vector_source
            if self.embedding_model is None:
                raise ValueError("An embedding_model is required for search")
            if self.vector_source is None:
                raise ValueError("A vector_source is required for search")

            query_vector = self.embedding_model.embed_text(query)

            if mode == "vector":
                raw = _normalize_knn_hits(self.vector_source.knn(
                    index_name=target,
                    query_vector=query_vector,
                    k=k,
                    filter_expr=_build_filter_expr(filters),
                    return_fields=_SEARCH_RETURN_FIELDS,
                ))
            else:
                # hybrid: RRF fusion of vector KNN + BM25
                if not self._has_indexed_chunks(target):
                    # Cold index → no BM25 ranks to fuse; this is still a useful
                    # vector-only result, but warn so the degradation isn't silent
                    # (distinct from bm25/rerank which return nothing — #50).
                    self._warn_cold_index("hybrid", target)
                bm25_scores = self._bm25_candidates(query, target, filters)
                if not bm25_scores:
                    raw = _normalize_knn_hits(self.vector_source.knn(
                        index_name=target,
                        query_vector=query_vector,
                        k=k,
                        filter_expr=_build_filter_expr(filters),
                        return_fields=_SEARCH_RETURN_FIELDS,
                    ))
                else:
                    k_vec = max(k * 3, 20)
                    vec_results = _normalize_knn_hits(self.vector_source.knn(
                        index_name=target,
                        query_vector=query_vector,
                        k=k_vec,
                        filter_expr=_build_filter_expr(filters),
                        return_fields=_SEARCH_RETURN_FIELDS,
                    ))
                    vec_rank: Dict[str, int] = {
                        r.get("chunk_id", ""): i + 1
                        for i, r in enumerate(vec_results)
                        if r.get("chunk_id")
                    }
                    bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
                    bm25_rank: Dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(bm25_ranked)}
                    rrf_k = 60
                    n_vec_fallback = len(vec_results) + 1
                    n_bm25_fallback = len(bm25_ranked) + 1
                    all_ids = set(vec_rank) | set(bm25_rank)
                    scored: List[Tuple[float, str]] = [
                        (
                            1.0 / (rrf_k + vec_rank.get(cid, n_vec_fallback))
                            + 1.0 / (rrf_k + bm25_rank.get(cid, n_bm25_fallback)),
                            cid,
                        )
                        for cid in all_ids
                    ]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    vec_by_id = {r.get("chunk_id", ""): r for r in vec_results}
                    raw = []
                    for rrf_score, chunk_id in scored[:k]:
                        if chunk_id in vec_by_id:
                            chunk = dict(vec_by_id[chunk_id])
                        elif chunk_id in self._chunk_index:
                            chunk = dict(self._chunk_index[chunk_id])
                        else:
                            continue
                        chunk["score"] = rrf_score
                        raw.append(chunk)

        if not rich:
            return raw

        # ------------------------------------------------------------------
        # Rich wrapping — provenance, neighbor context, by_document grouping
        # ------------------------------------------------------------------

        # Neighbor context reads adjacent chunks from the in-process index; when
        # it is cold (restart / separate process) context is silently omitted.
        # Warn once — but only when there are results to annotate (#50).
        if raw and context_chunks > 0 and not self._has_indexed_chunks(target):
            self._warn_cold_index("context", target)

        rich_results: List[Dict[str, Any]] = []
        for rank, chunk in enumerate(raw, 1):
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
            document_id = chunk.get("document_id", "")
            idx = self._chunk_idx_from_id(chunk_id)

            provenance: Dict[str, Any] = {
                "document_id": document_id,
                "file_name": chunk.get("file_name", ""),
                "chunk_id": chunk_id,
            }
            if idx is not None:
                provenance["chunk_index"] = idx
            if chunk.get("page_start") is not None:
                provenance["page"] = chunk["page_start"]
            if chunk.get("section_title"):
                provenance["section"] = chunk["section_title"]

            result: Dict[str, Any] = {
                "rank": rank,
                "score": chunk.get("score"),
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "provenance": provenance,
            }

            doc_type = chunk.get("document_type")
            if doc_type:
                result["document_type"] = doc_type

            if context_chunks > 0:
                context: Dict[str, str] = {}
                before_parts = [
                    t for i in range(context_chunks, 0, -1)
                    if (t := self._neighbor_text(chunk_id, -i)) is not None
                ]
                after_parts = [
                    t for i in range(1, context_chunks + 1)
                    if (t := self._neighbor_text(chunk_id, i)) is not None
                ]
                if before_parts:
                    context["before"] = " ".join(before_parts)
                if after_parts:
                    context["after"] = " ".join(after_parts)
                if context:
                    result["context"] = context

            rich_results.append(result)

        by_document: Dict[str, Dict[str, Any]] = {}
        for r in rich_results:
            doc_id = r["provenance"]["document_id"]
            if doc_id not in by_document:
                entry: Dict[str, Any] = {
                    "document_id": doc_id,
                    "file_name": r["provenance"].get("file_name", ""),
                    "match_count": 0,
                    "matches": [],
                }
                if r.get("document_type"):
                    entry["document_type"] = r["document_type"]
                by_document[doc_id] = entry
            by_document[doc_id]["match_count"] += 1
            by_document[doc_id]["matches"].append(r)

        for entry in by_document.values():
            seen: set = set()
            sections = []
            for m in entry["matches"]:
                sec = m["provenance"].get("section")
                if sec and sec not in seen:
                    sections.append(sec)
                    seen.add(sec)
            if sections:
                entry["top_sections"] = sections

        return {
            "query": query,
            "result_count": len(rich_results),
            "document_count": len(by_document),
            "results": rich_results,
            "by_document": by_document,
        }

    @staticmethod
    def _chunk_idx_from_id(chunk_id: str) -> Optional[int]:
        """Parse the numeric index from a chunk_id like ``doc_abc_chunk_000003``."""
        parts = chunk_id.rsplit("_chunk_", 1)
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return None

    def _neighbor_text(self, chunk_id: str, offset: int) -> Optional[str]:
        """Return the text of the chunk ``offset`` positions from ``chunk_id``."""
        idx = self._chunk_idx_from_id(chunk_id)
        if idx is None:
            return None
        doc_prefix = chunk_id.rsplit("_chunk_", 1)[0]
        neighbor_id = f"{doc_prefix}_chunk_{(idx + offset):06d}"
        neighbor = self._chunk_index.get(neighbor_id)
        return neighbor.get("text") if neighbor else None
