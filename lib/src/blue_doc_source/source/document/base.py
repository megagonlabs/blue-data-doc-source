from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Abstract interface for a vector store backend."""

    @abstractmethod
    def create_index(
        self,
        collection: str,
        index_name: str,
        dim: int,
    ) -> Any:
        """Create a vector index for a collection.

        Args:
            collection: Namespace or keyspace for the documents.
            index_name: Name of the vector index to create.
            dim: Embedding dimension.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        collection: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> Any:
        """Insert or update a document in the vector store.

        Args:
            collection: Namespace or keyspace for the document.
            doc_id: Unique identifier for the document.
            document: Dict that must include an embedding field.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the k nearest neighbours to a query vector.

        Args:
            index_name: Name of the vector index to query.
            query_vector: Embedding of the search query.
            k: Number of results to return.
            return_fields: Fields to include in each result dict.

        Returns:
            List of matching document dicts, ordered by similarity.
        """
        raise NotImplementedError