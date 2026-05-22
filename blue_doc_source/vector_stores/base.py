from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    @abstractmethod
    def create_index(
        self,
        collection: str,
        index_name: str,
        dim: int,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        collection: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError