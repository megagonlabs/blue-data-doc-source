import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from blue_data.source.vector.vector_source import VectorSource

from .chunkers import Chunker, SectionChunker
from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .chunk_record import ChunkRecord
from .parsers import DocxParser, Parser, PDFParser, TextParser
from .schema.base import DocumentClassifier, SchemaExtractor

# Chars that need escaping inside Redis Search TAG filter values.
_TAG_SPECIAL_RE = re.compile(r'([,.<>{}\[\]"\'`:;!@#$%^&*\-+=~|/ ])')

# Fields indexed as TAG in the vector index — used for pre-filtering.
_FILTER_TAG_FIELDS = ("document_id", "file_name", "section_title", "mime_type")

# Extra fields passed to create_index so Redis Search can filter on them.
_INDEX_EXTRA_FIELDS = [
    {"path": "$.document_id",   "name": "document_id",   "field_type": "TAG"},
    {"path": "$.file_name",     "name": "file_name",     "field_type": "TAG"},
    {"path": "$.section_title", "name": "section_title", "field_type": "TAG"},
    {"path": "$.mime_type",     "name": "mime_type",     "field_type": "TAG"},
    {"path": "$.page_start",    "name": "page_start",    "field_type": "NUMERIC"},
]


def _escape_redis_tag(value: str) -> str:
    """Escape special chars in a Redis Search TAG filter value."""
    return _TAG_SPECIAL_RE.sub(r'\\\1', value)


def _build_filter_expr(filters: Optional[Dict[str, Any]]) -> str:
    """Translate a filters dict into a Redis Search filter expression.

    Supported keys:
        document_id, file_name, section_title, mime_type  — exact TAG match
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
        schema_extractor: Optional[SchemaExtractor] = None,
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
        self.schema_extractor = schema_extractor
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap
        # path → (document_id, chunk_count, collection) — used to delete stale chunks
        self._doc_registry: Dict[str, Tuple[str, int, str]] = {}
        # collections for which create_index has already succeeded this session
        self._index_created: Set[str] = set()
        # document_id → schema record — populated during ingest when classifier is set
        self._schema_registry: Dict[str, Dict[str, Any]] = {}

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

        if self.parser is not None:
            pages = self.parser.parse(path)
            mime_type = "application/pdf" if suffix == ".pdf" else "text/plain"
            document_id = self._file_document_id(file_path)
            self._cleanup_stale_chunks(file_path, document_id)
            records = self._ingest_from_pages(
                pages=pages,
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
        if self.vector_source is None:
            return
        for idx in range(old_count):
            try:
                self.vector_source.delete(
                    collection=old_collection,
                    doc_id=f"{old_doc_id}_chunk_{idx:06d}",
                )
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
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []
        chunk_index = 0

        # Run document classification / schema extraction once per document.
        document_type: Optional[str] = None
        if self.classifier is not None:
            full_text = " ".join(p.get("text", "") for p in pages)
            document_type, confidence = self.classifier.classify(full_text)
            extracted_fields: Dict[str, Any] = {}
            if self.schema_extractor is not None:
                extracted_fields = self.schema_extractor.extract(full_text, document_type)
            classifier_name = (
                "llm" if self.schema_extractor is not None else "keyword"
            )
            self._schema_registry[document_id] = {
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
                self._upsert(chunk_id, record_dict, collection)
                chunk_index += 1

        return records

    def get_schema(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return the schema record for a document, or ``None`` if not available.

        Schema records are populated during ingest when a ``classifier`` is set.
        They are stored in-process only — cross-process retrieval is not supported.
        """
        return self._schema_registry.get(document_id)

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
            collection: Collection to delete from. Defaults to the registered
                collection for the path, or the instance-level default.

        Returns:
            Number of successful delete calls issued.

        Raises:
            ValueError: If neither ``path`` nor ``document_id + chunk_count`` is
                provided, if ``path`` is not in the registry, or if
                ``vector_source`` is not set.
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
            doc_id, count, reg_collection = self._doc_registry[key]
            target = collection or reg_collection
        elif document_id is not None and chunk_count is not None:
            doc_id = document_id
            count = chunk_count
            target = collection or self.collection
        else:
            raise ValueError(
                "Provide either 'path' or both 'document_id' and 'chunk_count'"
            )

        deleted = 0
        for idx in range(count):
            chunk_id = f"{doc_id}_chunk_{idx:06d}"
            try:
                self.vector_source.delete(collection=target, doc_id=chunk_id)
                deleted += 1
            except Exception:
                pass

        if path is not None:
            self._doc_registry.pop(str(Path(path)), None)

        return deleted

    def search(
        self,
        query: str,
        k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Embed a query and return the k nearest chunks from the vector store.

        Args:
            query: Natural language query string.
            k: Number of results to return.
            collection: Override the instance-level collection for this call only.
            filters: Optional pre-filter applied before KNN. Supported keys:
                ``document_id``, ``file_name``, ``section_title``, ``mime_type``
                (exact string match) and ``page_start`` (int for exact page, or
                ``{"gte": N, "lte": M}`` for a range). Requires ``ensure_index``
                to have been called after this feature was added; existing indexes
                without the extra filterable fields must be recreated.

        Returns:
            List of matching chunk dicts as returned by the vector store.

        Raises:
            ValueError: If ``embedding_model`` or ``vector_source`` is not set.
        """
        if self.embedding_model is None:
            raise ValueError("An embedding_model is required for search")
        if self.vector_source is None:
            raise ValueError("A vector_source is required for search")

        query_vector = self.embedding_model.embed_text(query)
        return self.vector_source.knn(
            index_name=collection or self.collection,
            query_vector=query_vector,
            k=k,
            filter_expr=_build_filter_expr(filters),
        )
