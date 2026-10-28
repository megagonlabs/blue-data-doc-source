"""Request-model bounds for the doc API (#80 / DOCSRC-12).

Caller-supplied ``k``, ``context_chunks`` and ``chunk_count`` are unbounded in
the library unless validated, and the REST models used to pass them through
unchanged — one request could drive linear CPU (the rich-context loop) or
backend round-trips (the delete loop). The request models now carry
``Field(ge=, le=)`` constraints, so out-of-range values are rejected with 422
(pydantic validation) before any handler runs. Auth is disabled here so the
request clears the key guard and reaches model validation.
"""
import pytest
from fastapi.testclient import TestClient

from doc_api.main import app


@pytest.fixture()
def _auth_disabled(monkeypatch):
    monkeypatch.setenv("BLUE_DOC_API_AUTH_DISABLED", "true")
    monkeypatch.delenv("BLUE_DOC_ROOT", raising=False)


@pytest.mark.parametrize("k", [0, -1, 101, 10**9])
def test_search_k_out_of_range_rejected(_auth_disabled, k):
    with TestClient(app) as c:
        r = c.post("/documents/search", json={"query": "x", "k": k})
    assert r.status_code == 422


@pytest.mark.parametrize("n", [-1, 6, 10**6])
def test_search_context_chunks_out_of_range_rejected(_auth_disabled, n):
    with TestClient(app) as c:
        r = c.post("/documents/search", json={"query": "x", "context_chunks": n})
    assert r.status_code == 422


@pytest.mark.parametrize("chunk_count", [0, -1, 100001, 10**9])
def test_delete_chunk_count_out_of_range_rejected(_auth_disabled, chunk_count):
    with TestClient(app) as c:
        r = c.request(
            "DELETE",
            "/documents/document",
            json={"document_id": "doc_x", "chunk_count": chunk_count},
        )
    assert r.status_code == 422


@pytest.mark.parametrize("k", [0, -1, 101])
def test_find_then_search_k_out_of_range_rejected(_auth_disabled, k):
    with TestClient(app) as c:
        r = c.post("/documents/find-then-search", json={"query": "x", "k": k})
    assert r.status_code == 422
