"""Regression tests for issue #126.

DOCUMENT.INGEST accepted an arbitrary server path. With a configured documents
root, caller-supplied ingest paths must resolve under that root (via blue-data's
contain()) and anything outside — traversal, absolute paths, or symlinks escaping
the root — is refused. With no root configured, behaviour is unchanged.
"""
import os

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source.source.document.document_source import DocumentDataSource
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel


def _source(root=None):
    kwargs = {"embedding_model": DummyEmbeddingModel(dim=8), "vector_source": MemoryVectorSource(name="m")}
    if root is not None:
        kwargs["documents_root"] = str(root)
    src = DocumentDataSource(name="d", **kwargs)
    src.execute_query({"op": "DOCUMENT.ENSURE_INDEX"})
    return src


def _ingest(src, path):
    return src.execute_query({"op": "DOCUMENT.INGEST", "path": path})


def test_accepts_path_inside_root(tmp_path):
    root = tmp_path / "root"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "policy.txt").write_text("hello world from the policy", encoding="utf-8")

    out = _ingest(_source(root), "reports/policy.txt")
    assert isinstance(out, list) and len(out) >= 1


@pytest.mark.parametrize("bad", ["../outside/secret.txt", ".."])
def test_refuses_traversal(tmp_path, bad):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        _ingest(_source(root), bad)


def test_refuses_absolute_path_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        _ingest(_source(root), str(secret))


def test_refuses_symlink_escaping_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    os.symlink(outside, root / "link")

    with pytest.raises(ValueError):
        _ingest(_source(root), "link/secret.txt")


def test_no_root_is_unconfined(tmp_path):
    # Historical behaviour: without a documents root, an absolute path is accepted.
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")

    out = _ingest(_source(root=None), str(doc))
    assert isinstance(out, list) and len(out) >= 1
