from typing import Any, Dict, List, Optional

from .base import VectorStore


class RedisVectorAdapter(VectorStore):
    """
    Adapter around Blue's existing RedisVectorSource.

    This keeps the document source independent from Redis-specific
    implementation details while still reusing Blue's vector infrastructure.
    """

    def __init__(
        self,
        redis_source: Any,
        default_dim: int = 1536,
        embedding_field: str = "embedding",
    ):
        self.redis_source = redis_source
        self.default_dim = default_dim
        self.embedding_field = embedding_field

    def create_index(
        self,
        collection: str,
        index_name: str,
        dim: Optional[int] = None,
    ) -> Any:
        return self.redis_source.execute_query(
            {
                "op": "VECTOR.CREATE_INDEX",
                "index": index_name,
                "collection": collection,
                "dim": dim or self.default_dim,
                "vector_field": self.embedding_field,
                "vector_path": f"$.{self.embedding_field}",
            }
        )

    def upsert(
        self,
        collection: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> Any:
        if self.embedding_field not in document:
            raise ValueError(
                f"Document must contain embedding field '{self.embedding_field}'"
            )

        return self.redis_source.execute_query(
            {
                "op": "VECTOR.UPSERT",
                "collection": collection,
                "id": doc_id,
                "doc": document,
                "embedding_field": self.embedding_field,
                "index_dimension": self.default_dim,
            }
        )

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.redis_source.execute_query(
            {
                "op": "VECTOR.KNN",
                "index": index_name,
                "query_vector": query_vector,
                "k": k,
                "vector_field": self.embedding_field,
                "return_fields": return_fields
                or [
                    "text",
                    "document_id",
                    "chunk_id",
                    "file_name",
                    "page_start",
                    "page_end",
                ],
            }
        )