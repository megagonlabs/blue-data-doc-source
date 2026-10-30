"""Every DOCUMENT.* op is declared a read or a write, and blue-data's registry
refuses the writes without ``commit`` (issue #170)."""
import inspect
import re

import pytest
from blue_data.registry.data_registry import DataRegistry

from blue_data_doc_source.source.document.document_source import DocumentDataSource


def _served_ops():
    return set(re.findall(r'op == "(DOCUMENT\.\w+)"', inspect.getsource(DocumentDataSource.execute_query)))


def test_every_served_op_is_in_exactly_one_set():
    served = _served_ops()
    assert served
    assert not DocumentDataSource._READ_OPS & DocumentDataSource._WRITE_OPS
    assert DocumentDataSource._READ_OPS | DocumentDataSource._WRITE_OPS == served


def test_write_ops_are_the_state_changing_ones():
    assert DocumentDataSource._WRITE_OPS == {"DOCUMENT.INGEST", "DOCUMENT.DELETE", "DOCUMENT.ENSURE_INDEX"}


@pytest.fixture
def registry(tmp_path):
    (tmp_path / "placeholder.txt").write_text("Placeholder document for a read-only check.\n")
    src = DocumentDataSource("docs", properties={
        "embedding": "dummy", "vector": "memory", "collection": "demo", "documents_root": str(tmp_path),
    })
    reg = DataRegistry()
    reg.register_source(src)
    return reg, src


@pytest.mark.parametrize("query", [
    {"op": "DOCUMENT.ENSURE_INDEX"},
    {"op": "DOCUMENT.INGEST", "path": "placeholder.txt"},
    {"op": "DOCUMENT.DELETE", "document_id": "x", "chunk_count": 1},
])
def test_write_op_without_commit_is_refused(registry, query):
    reg, _ = registry
    with pytest.raises(ValueError, match="commit"):
        reg.execute_query("docs", query)


def test_ingest_without_commit_writes_nothing(registry):
    reg, src = registry
    src.execute_query({"op": "DOCUMENT.ENSURE_INDEX"})
    with pytest.raises(ValueError, match="commit"):
        reg.execute_query("docs", {"op": "DOCUMENT.INGEST", "path": "placeholder.txt"})
    assert src.execute_query({"op": "DOCUMENT.SEARCH", "query": "placeholder", "k": 1}) == []


def test_write_then_read_with_and_without_commit(registry):
    reg, _ = registry
    reg.execute_query("docs", {"op": "DOCUMENT.ENSURE_INDEX"}, optional_properties={"commit": True})
    reg.execute_query("docs", {"op": "DOCUMENT.INGEST", "path": "placeholder.txt"},
                      optional_properties={"commit": True})
    hits = reg.execute_query("docs", {"op": "DOCUMENT.SEARCH", "query": "placeholder", "k": 1})
    assert len(hits) == 1
