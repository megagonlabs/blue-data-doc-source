# blue_doc_source/pipeline/document_pipeline.py

from pathlib import Path
from uuid import uuid4
from typing import Dict, List, Any

from blue_doc_source.chunkers import WindowChunker
from blue_doc_source.models import ChunkRecord
from blue_doc_source.parsers import TextParser


class DocumentPipeline:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.parser = TextParser()
        self.chunker = WindowChunker(chunk_size=chunk_size, overlap=overlap)
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

            record = ChunkRecord(
                document_id=document_id,
                chunk_id=chunk_id,
                file_name=file_path.name,
                text=chunk_text,
                metadata={
                    "source_path": str(file_path),
                    "chunk_index": idx,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                },
            )

            records.append(record.to_dict())

        return records