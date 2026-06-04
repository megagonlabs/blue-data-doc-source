from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from blue_doc_source.chunkers import WindowChunker
from blue_doc_source.embeddings import EmbeddingModel
from blue_doc_source.models import ChunkRecord
from blue_doc_source.parsers import TextParser


class DocumentPipeline:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.parser = TextParser()
        self.chunker = WindowChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, path: str) -> List[Dict[str, Any]]:
        file_path = Path(path)
        text = self.parser.parse(path)
        chunks = self.chunker.chunk(text)

        document_id = f"doc_{uuid4().hex}"
        records = []

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx:04d}"

            embedding = None
            if self.embedding_model is not None:
                embedding = self.embedding_model.embed_text(chunk_text)

            record = ChunkRecord(
                document_id=document_id,
                chunk_id=chunk_id,
                file_name=file_path.name,
                text=chunk_text,
                embedding=embedding,
                metadata={
                    "source_path": str(file_path),
                    "chunk_index": idx,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "embedding_model": (
                        self.embedding_model.__class__.__name__
                        if self.embedding_model is not None
                        else None
                    ),
                },
            )

            records.append(record.to_dict())

        return records