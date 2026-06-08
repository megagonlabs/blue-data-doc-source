from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from blue_doc_source.chunkers import WindowChunker
from blue_doc_source.embeddings import EmbeddingModel
from blue_doc_source.enrichment import Enricher
from blue_doc_source.models import ChunkRecord
from blue_doc_source.parsers import PDFParser, TextParser
from blue_doc_source.vector_stores import VectorStore


class DocumentPipeline:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        enricher: Optional[Enricher] = None,
        collection: str = "documents",
    ):
        self.text_parser = TextParser()
        self.pdf_parser = PDFParser()
        self.chunker = WindowChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.enricher = enricher
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, path: str) -> List[Dict[str, Any]]:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return self._ingest_txt(file_path)
        elif suffix == ".pdf":
            return self._ingest_pdf(file_path)
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
        mime_type: str = "text/plain",
    ) -> ChunkRecord:
        embedding = None
        if self.embedding_model is not None:
            embedding = self.embedding_model.embed_text(chunk_text)

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

    def _upsert(self, chunk_id: str, record_dict: Dict[str, Any]) -> None:
        if self.vector_store is not None and record_dict.get("embedding") is not None:
            self.vector_store.upsert(
                collection=self.collection,
                doc_id=chunk_id,
                document=record_dict,
            )

    def _ingest_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        text = self.text_parser.parse(str(file_path))
        chunks = self.chunker.chunk(text)

        document_id = f"doc_{uuid4().hex}"
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx:04d}"
            record = self._make_record(
                document_id=document_id,
                chunk_id=chunk_id,
                file_path=file_path,
                chunk_text=chunk_text,
                chunk_index=idx,
                mime_type="text/plain",
                ingested_at=ingested_at,
            )
            record_dict = record.to_dict()
            records.append(record_dict)
            self._upsert(chunk_id, record_dict)

        return records

    def _ingest_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        pages = self.pdf_parser.parse_pages(str(file_path))

        document_id = f"doc_{uuid4().hex}"
        ingested_at = datetime.now(timezone.utc).isoformat()
        records = []
        chunk_index = 0

        for page in pages:
            page_num = page["page_num"]
            chunks = self.chunker.chunk(page["text"])

            for chunk_text in chunks:
                chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
                record = self._make_record(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    file_path=file_path,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    page_start=page_num,
                    page_end=page_num,
                    mime_type="application/pdf",
                    ingested_at=ingested_at,
                )
                record_dict = record.to_dict()
                records.append(record_dict)
                self._upsert(chunk_id, record_dict)
                chunk_index += 1

        return records

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.embedding_model is None:
            raise ValueError("An embedding_model is required for search")
        if self.vector_store is None:
            raise ValueError("A vector_store is required for search")

        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(
            index_name=self.collection,
            query_vector=query_vector,
            k=k,
        )
