"""Demo-only LocalRedisVectorStore — not part of the blue_doc_source package.

Requires Redis Stack running locally:
    docker run -p 6379:6379 redis/redis-stack-server:latest
"""
import struct
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from blue_doc_source.vector_stores import VectorStore


class LocalRedisVectorStore(VectorStore):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.client.ping()

    def create_index(self, collection: str, index_name: str, dim: Optional[int] = None) -> Any:
        dim = dim or 1536
        schema = (
            TextField("$.text", as_name="text"),
            TextField("$.document_id", as_name="document_id"),
            TextField("$.chunk_id", as_name="chunk_id"),
            TextField("$.file_name", as_name="file_name"),
            NumericField("$.page_start", as_name="page_start"),
            NumericField("$.page_end", as_name="page_end"),
            VectorField(
                "$.embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                as_name="embedding",
            ),
        )
        definition = IndexDefinition(prefix=[f"{collection}:"], index_type=IndexType.JSON)
        try:
            self.client.ft(index_name).create_index(schema, definition=definition)
        except Exception as e:
            if "Index already exists" not in str(e):
                raise

    def upsert(self, collection: str, doc_id: str, document: Dict[str, Any]) -> Any:
        self.client.json().set(f"{collection}:{doc_id}", "$", document)

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        fields = return_fields or ["text", "document_id", "chunk_id", "file_name", "page_start", "page_end"]
        vec_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)
        q = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields(*fields, "score")
            .dialect(2)
        )
        results = self.client.ft(index_name).search(q, query_params={"vec": vec_bytes})
        out = []
        for doc in results.docs:
            row = {}
            for f in [*fields, "score"]:
                val = getattr(doc, f, None)
                if val is not None:
                    row[f] = val.decode() if isinstance(val, bytes) else val
            out.append(row)
        return out
