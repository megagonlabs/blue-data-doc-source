"""Demo-only LocalRedisVectorStore — not part of the blue_data_doc_source package.

Requires Redis Stack running locally:
    docker run -p 6379:6379 redis/redis-stack-server:latest
"""
import json
import struct
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class LocalRedisVectorStore:
    """Standalone Redis vector store for demo purposes.

    Implements the same upsert/knn interface as blue-data's VectorSource so it
    can be passed directly as the vector_source argument to DocumentPipeline.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.client.ping()

    def create_index(
        self,
        index_name: str,
        collection: str,
        dim: int = 1536,
        distance_metric: str = "COSINE",
        extra_fields: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        _field_type_map = {"TAG": TagField, "NUMERIC": NumericField, "TEXT": TextField}
        extra_schema = []
        for f in (extra_fields or []):
            cls = _field_type_map.get(f.get("field_type", "TEXT"), TextField)
            extra_schema.append(cls(f["path"], as_name=f["name"]))

        schema = (
            TextField("$.text", as_name="text"),
            TextField("$.chunk_id", as_name="chunk_id"),
            VectorField(
                "$.embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": distance_metric},
                as_name="embedding",
            ),
            *extra_schema,
        )
        definition = IndexDefinition(prefix=[f"{collection}:"], index_type=IndexType.JSON)

        # Drop the existing index so a dim change always takes effect.
        try:
            self.client.ft(index_name).dropindex()
        except Exception:
            pass
        self.client.ft(index_name).create_index(schema, definition=definition)

    def upsert(
        self,
        collection: str,
        doc_id: str,
        doc: Dict[str, Any],
        embedding_field: str = "embedding",
    ) -> Any:
        self.client.json().set(f"{collection}:{doc_id}", "$", doc)

    def knn(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        embedding_field: str = "embedding",
        score_alias: str = "score",
        return_fields: Optional[List[str]] = None,
        filter_expr: str = "*",
    ) -> List[Dict[str, Any]]:
        vec_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)
        q = (
            Query(f"{filter_expr}=>[KNN {k} @{embedding_field} $vec AS {score_alias}]")
            .sort_by(score_alias)
            .paging(0, k)
            .return_fields("$", score_alias)
            .dialect(2)
        )
        results = self.client.ft(index_name).search(q, query_params={"vec": vec_bytes})
        out = []
        for doc in results.docs:
            raw_json = getattr(doc, "$", None) or getattr(doc, "json", None)
            if raw_json is None:
                continue
            chunk = json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
            chunk[score_alias] = float(getattr(doc, score_alias, 0))
            chunk.pop("embedding", None)
            out.append(chunk)
        return out

    def delete(self, collection: str, doc_id: str) -> None:
        self.client.delete(f"{collection}:{doc_id}")
