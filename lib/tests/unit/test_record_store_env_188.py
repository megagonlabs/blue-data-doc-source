"""Regression tests for issue #188.

The REST and MCP pipelines kept DocumentRecords in the default MemoryRecordStore, so a
restart lost FIND / FIND_THEN_SEARCH / GET_RECORD / path-delete while the chunks stayed
in Redis. ``record_store_from_env`` now returns a persistent store: the SAME Redis as the
chunks when one is configured, else a file.
"""
import os

from blue_data_doc_source import record_store_from_env
from blue_data_doc_source.source.document.schema import FileRecordStore, JsonSourceRecordStore


def test_no_redis_uses_a_file_store(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("BLUE_DOC_REDIS"):
            monkeypatch.delenv(k, raising=False)
    path = tmp_path / "records.json"
    monkeypatch.setenv("BLUE_DOC_RECORD_STORE", str(path))
    store = record_store_from_env("BLUE_DOC")
    assert isinstance(store, FileRecordStore)

    # it actually persists: a new store at the same path sees the record
    store.upsert({"document_id": "doc_1", "document_type": "contract", "chunk_count": 3})
    assert record_store_from_env("BLUE_DOC").get("doc_1")["chunk_count"] == 3


def test_redis_env_uses_a_redis_backed_store(monkeypatch):
    # redis.Redis() is lazy, so no server is contacted at construction.
    monkeypatch.setenv("BLUE_DOC_REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("BLUE_DOC_REDIS_PORT", "6399")
    store = record_store_from_env("BLUE_DOC")
    assert isinstance(store, JsonSourceRecordStore)
    assert not isinstance(store, FileRecordStore)  # a Redis-backed store, not a file


def test_redis_url_also_selects_redis(monkeypatch):
    for k in ("HOST", "PORT"):
        monkeypatch.delenv(f"BLUE_DOC_REDIS_{k}", raising=False)
    monkeypatch.setenv("BLUE_DOC_REDIS_URL", "redis://:pw@r.example:6380/0")
    store = record_store_from_env("BLUE_DOC")
    assert isinstance(store, JsonSourceRecordStore) and not isinstance(store, FileRecordStore)
