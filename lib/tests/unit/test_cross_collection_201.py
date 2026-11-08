"""Regression tests for issue #201.

Chunk ids come from the file's bytes, so the same file ingested into two collections
shares chunk ids. The in-process maps keyed those by chunk id / path alone, so the
second ingest overwrote the first: BM25 in the first collection returned nothing, and a
path delete reached only the last collection, leaving the other copy behind. The maps are
now keyed so both collections are independently searchable and deletable.
"""
import tempfile
from pathlib import Path

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source import DocumentPipeline
from blue_data_doc_source.source.document.embeddings.dummy_embedding import DummyEmbeddingModel

TEXT = " ".join(f"Sentence {i} covers the termination and payment terms." for i in range(30))


@pytest.fixture
def two_collections():
    f = Path(tempfile.mkdtemp()) / "a.txt"
    f.write_text(TEXT)
    p = DocumentPipeline(chunk_size=60, overlap=0, embedding_model=DummyEmbeddingModel(),
                         vector_source=MemoryVectorSource(name="m"), collection="c1")
    p.ensure_index("c1")
    p.ensure_index("c2")
    p.ingest(str(f), collection="c1")
    p.ingest(str(f), collection="c2")
    return p, str(f)


def _n(p, collection, mode):
    return len(p.search("termination", mode=mode, k=50, collection=collection))


def test_bm25_finds_chunks_in_both_collections(two_collections):
    p, _ = two_collections
    # Pre-fix: bm25 in the first collection returned 0 (its chunk→collection map was
    # overwritten by the second ingest).
    assert _n(p, "c1", "bm25") > 0
    assert _n(p, "c1", "bm25") == _n(p, "c2", "bm25")


def test_chunk_belongs_to_both_collections(two_collections):
    p, _ = two_collections
    # The same chunk id is tracked as belonging to BOTH collections (a set, not the last).
    assert all({"c1", "c2"} <= cols for cols in p._chunk_collection.values())


def test_delete_path_removes_every_collection_copy(two_collections):
    p, path = two_collections
    deleted = p.delete(path=path)  # no collection -> both copies
    assert deleted == 8  # 4 chunks * 2 collections
    assert _n(p, "c1", "vector") == 0
    assert _n(p, "c2", "vector") == 0


def test_delete_path_scoped_to_one_collection(two_collections):
    p, path = two_collections
    deleted = p.delete(path=path, collection="c1")  # only c1
    assert deleted == 4
    assert _n(p, "c1", "vector") == 0
    assert _n(p, "c2", "vector") == 4  # the other copy survives


def test_delete_path_bogus_collection_falls_back_to_stored(two_collections):
    # #78: a collection the file was never stored in must not redirect the delete away
    # from where it actually lives.
    p, path = two_collections
    deleted = p.delete(path=path, collection="never-stored")
    assert deleted == 8
    assert _n(p, "c1", "vector") == 0 and _n(p, "c2", "vector") == 0
