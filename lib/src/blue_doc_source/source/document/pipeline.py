from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from blue_data.source.vector.vector_source import VectorSource

from .chunkers import Chunker, SectionChunker
from .embeddings import EmbeddingModel
from .enrichment import Enricher
from .chunk_record import ChunkRecord
from .parsers import PDFParser, TextParser


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
        embedding_model: Optional[EmbeddingModel] = None,
        vector_source: Optional[VectorSource] = None,
        enricher: Optional[Enricher] = None,
        collection: str = "documents",
    ):
        """
        Args:
            chunk_size: Maximum words per chunk. Passed to the default SectionChunker.
            overlap: Word overlap between consecutive chunks.
            chunker: Chunker instance to use. Defaults to ``SectionChunker``, which
                detects section headers and falls back to window chunking.
            embedding_model: Embedding model to use. When ``None``, no
                embeddings are generated and upsert is skipped.
            vector_source: VectorSource backend from blue-data. When ``None``,
                chunks are returned but not persisted.
            enricher: Enrichment step. When ``None``, ``entities`` and
                ``semantic_hints`` are left empty on each record.
            collection: Default collection name used when upserting documents.
        """
        self.text_parser = TextParser()
        self.pdf_parser = PDFParser()
        self.chunker = chunker or SectionChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = embedding_model
        self.vector_source = vector_source
        self.enricher = enricher
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

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

        if suffix == ".txt":
            return self._ingest_txt(file_path, effective_collection)
        elif suffix == ".pdf":
            return self._ingest_pdf(file_path, effective_collection)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _make_record(
        self,
        document_id: str,
        chunk_id: str,
        file_path: Path,
        chunk_text: str,
        chunk_index: int,
        ingested_at: str,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        section_title: Optional[str] = None,
        mime_type: str = "text/plain",
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

        return ChunkRecord(
            document_id=document_id,
            chunk_id=chunk_id,
            file_name=file_path.name,
            mime_type=mime_type,
            text=chunk_text,
            page_start=page_start,
            page_end=page_end,
            section_title=section_title,
            entities=entities,
            semantic_hints=semantic_hints,
            embedding=embedding,
            metadata={
                "source_path": str(file_path),
                "chunk_index": chunk_index,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "ingested_at": ingested_at,
                "embedding_model": (
                    self.embedding_model.__class__.__name__
                    if self.embedding_model is not None
                    else None
                ),
            },
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

        Must be called before the first ingest when using a vector store, or the
        store may reject upserts that target a non-existent index.

        Args:
            collection: Collection / index name. Defaults to the instance-level collection.
        """
        if self.vector_source is None or self.embedding_model is None:
            return
        self.vector_source.create_index(
            index_name=collection or self.collection,
            collection=collection or self.collection,
            dim=self.embedding_model.dim,
        )

    def _ingest_txt(self, file_path: Path, collection: str) -> List[Dict[str, Any]]:
        text = self.text_parser.parse(str(file_path))
        chunks = self.chunker.chunk_with_titles(text)

        document_id = self._file_document_id(file_path)
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []

        for idx, (section_title, chunk_text) in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx:04d}"
            record = self._make_record(
                document_id=document_id,
                chunk_id=chunk_id,
                file_path=file_path,
                chunk_text=chunk_text,
                chunk_index=idx,
                section_title=section_title,
                mime_type="text/plain",
                ingested_at=ingested_at,
            )
            record_dict = record.to_dict()
            records.append(record_dict)
            self._upsert(chunk_id, record_dict, collection)

        return records

    def _ingest_pdf(self, file_path: Path, collection: str) -> List[Dict[str, Any]]:
        pages = self.pdf_parser.parse_pages(str(file_path))

        document_id = self._file_document_id(file_path)
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []
        chunk_index = 0

        for page in pages:
            page_num = page["page_num"]
            chunks = self.chunker.chunk_with_titles(page["text"])

            for section_title, chunk_text in chunks:
                chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
                record = self._make_record(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    file_path=file_path,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    page_start=page_num,
                    page_end=page_num,
                    section_title=section_title,
                    mime_type="application/pdf",
                    ingested_at=ingested_at,
                )
                record_dict = record.to_dict()
                records.append(record_dict)
                self._upsert(chunk_id, record_dict, collection)
                chunk_index += 1

        return records

    def search(self, query: str, k: int = 5, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        """Embed a query and return the k nearest chunks from the vector store.

        Args:
            query: Natural language query string.
            k: Number of results to return.
            collection: Override the instance-level collection for this call only.

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
        )
