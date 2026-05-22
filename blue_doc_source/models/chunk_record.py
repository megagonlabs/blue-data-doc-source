from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkRecord:
    document_id: str
    chunk_id: str
    text: str
    file_name: str
    source_type: str = "document"
    mime_type: str = "text/plain"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    semantic_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "text": self.text,
            "entities": self.entities,
            "semantic_hints": self.semantic_hints,
            "metadata": self.metadata,
        }

        if self.page_start is not None:
            result["page_start"] = self.page_start

        if self.page_end is not None:
            result["page_end"] = self.page_end

        if self.section_title is not None:
            result["section_title"] = self.section_title

        if self.embedding is not None:
            result["embedding"] = self.embedding

        return result