"""Demo-only LocalRedisVectorStore — not part of the blue_doc_source package.

Requires Redis Stack running locally:
    docker run -p 6379:6379 redis/redis-stack-server:latest
"""
import json
import struct
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType


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
    ) -> Any:
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
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": distance_metric},
                as_name="embedding",
            ),
        )
        definition = IndexDefinition(prefix=[f"{collection}:"], index_type=IndexType.JSON)
        try:
            self.client.ft(index_name).create_index(schema, definition=definition)
        except Exception as e:
            if "Index already exists" not in str(e):
                raise

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
        raw = self.client.execute_command(
            "FT.SEARCH", index_name,
            f"*=>[KNN {k} @{embedding_field} $vec AS {score_alias}]",
            "PARAMS", "2", "vec", vec_bytes,
            "SORTBY", score_alias,
            "DIALECT", "2",
        )
        results = raw.get(b"results", [])
        out = []
        for item in results:
            extra = item.get(b"extra_attributes", {})
            score = extra.get(b"score", b"0")
            raw_doc = extra.get(b"$", b"{}")
            doc = json.loads(raw_doc)
            doc[score_alias] = float(score)
            doc.pop("embedding", None)
            out.append(doc)
        return out
