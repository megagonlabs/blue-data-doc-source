"""doc-api config tests — no external services required.

`RedisVectorSource` constructs lazily (no socket opened at __init__), so the
Redis-branch regression test below stays fully offline.
"""
from __future__ import annotations


def test_build_pipeline_with_redis_constructs_vector_source(monkeypatch):
    # Regression (#91): when BLUE_DOC_REDIS_HOST is set, the builder must construct
    # a RedisVectorSource WITHOUT raising TypeError — connection details go in
    # `properties={"connection": {...}}`, not as host/port kwargs.
    from doc_api.config import _build_pipeline

    monkeypatch.setenv("BLUE_DOC_REDIS_HOST", "localhost")
    monkeypatch.setenv("BLUE_DOC_REDIS_PORT", "6379")
    pipeline = _build_pipeline()
    assert type(pipeline.vector_source).__name__ == "RedisVectorSource"


def test_build_pipeline_without_redis_has_no_vector_source(monkeypatch):
    # The default (no Redis configured) path returns a pipeline with no vector store.
    from doc_api.config import _build_pipeline

    monkeypatch.delenv("BLUE_DOC_REDIS_HOST", raising=False)
    pipeline = _build_pipeline()
    assert pipeline.vector_source is None
