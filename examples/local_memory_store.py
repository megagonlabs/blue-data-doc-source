"""Demo-only LocalMemoryVectorStore — not part of the blue_data_doc_source package.

In-process vector store backed by numpy cosine similarity.
No external services required. All data is lost when the process exits.

Use this for local experimentation and running demos without Redis.
For persistent, cross-process storage see examples/local_redis_store.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class LocalMemoryVectorStore:
    """In-memory vector store for demos.

    Implements the create_index / upsert / delete / knn interface expected by
    DocumentPipeline and DocumentDataSource (same signature as blue-data's
    VectorSource abstract methods).

    Scores returned by knn are cosine similarity in [−1, 1]
    (higher = more relevant).  Compare to RedisVectorSource which returns
    cosine distance (lower = more relevant).
    """

    def __init__(self) -> None:
        # collection -> {doc_id -> full doc dict including embedding}
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # VectorSource interface                                               #
    # ------------------------------------------------------------------ #

    def create_index(
        self,
        index_name: str,
        collection: str,
        dim: int,
        distance_metric: str = "COSINE",
        extra_fields: list | None = None,
    ) -> str:
        self._store.setdefault(collection, {})
        return index_name

    def upsert(
        self,
        collection: str,
        doc_id: str,
        doc: dict[str, Any],
        embedding_field: str = "embedding",
    ) -> str:
        self._store.setdefault(collection, {})[doc_id] = dict(doc)
        return doc_id

    def delete(self, collection: str, doc_id: str) -> int:
        removed = self._store.get(collection, {}).pop(doc_id, None)
        return 1 if removed is not None else 0

    def knn(
        self,
        index_name: str,
        query_vector: list[float],
        k: int = 5,
        embedding_field: str = "embedding",
        score_alias: str = "score",
        return_fields: list[str] | None = None,
        filter_expr: str = "*",
    ) -> list[dict[str, Any]]:
        collection = index_name  # pipeline uses collection name as index name
        store = self._store.get(collection, {})
        if not store:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0.0:
            return []

        scored: list[tuple[float, str, dict]] = []
        for doc_id, doc in store.items():
            emb = doc.get(embedding_field)
            if emb is None:
                continue
            ev = np.array(emb, dtype=np.float32)
            en = np.linalg.norm(ev)
            if en == 0.0:
                continue
            sim = float(np.dot(qv, ev) / (qn * en))
            scored.append((sim, doc_id, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, _doc_id, doc in scored[:k]:
            row = {key: val for key, val in doc.items() if key != embedding_field}
            row[score_alias] = sim
            results.append(row)
        return results

    # ------------------------------------------------------------------ #
    # Inspection helpers (not part of the interface)                      #
    # ------------------------------------------------------------------ #

    def count(self, collection: str) -> int:
        """Return the number of docs stored in *collection*."""
        return len(self._store.get(collection, {}))

    def doc_ids(self, collection: str) -> list[str]:
        """Return all doc IDs stored in *collection*."""
        return list(self._store.get(collection, {}).keys())
