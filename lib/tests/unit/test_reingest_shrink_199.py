"""Regression tests for issue #199.

Re-ingesting the same file with a LARGER chunk size produces fewer chunks. The upsert
overwrites chunks [0, new) in place but left the old tail [new, old) in the store — still
searchable and, because the registry now recorded the smaller count, out of reach of
delete(path). The pipeline now trims that surplus tail at ingest time by asking the
vector store which chunks of the document it actually holds. No external services.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from blue_data.source.vector.memory_vector_source import MemoryVectorSource
from blue_data_doc_source import DocumentPipeline
from blue_data_doc_source.source.document.embeddings.dummy_embedding import DummyEmbeddingModel

TEXT = " ".join(f"Sentence number {i} talks about the deductible and premium amounts." for i in range(120))


def _file():
    f = Path(tempfile.mkdtemp()) / "policy.txt"
    f.write_text(TEXT)
    return f


def _stored(vs, collection):
    return len(vs.list_ids(collection))


def test_reingest_fewer_chunks_trims_the_tail_across_instances():
    # Two pipeline instances sharing only the vector store (the reported scenario):
    # the second has an empty registry / record store, so the fix must consult the store.
    f = _file()
    vs = MemoryVectorSource(name="m")
    emb, coll = DummyEmbeddingModel(), "c"
    small = DocumentPipeline(chunk_size=60, overlap=0, embedding_model=emb, vector_source=vs, collection=coll)
    small.ensure_index()
    n1 = len(small.ingest(str(f)))
    big = DocumentPipeline(chunk_size=400, overlap=0, embedding_model=emb, vector_source=vs, collection=coll)
    n2 = len(big.ingest(str(f)))

    assert n1 > n2  # fewer chunks the second time
    assert _stored(vs, coll) == n2  # the old tail is gone, not left behind
    assert big.delete(path=str(f)) == n2  # delete(path) now reaches every stored chunk
    assert _stored(vs, coll) == 0


def test_reingest_more_chunks_keeps_all():
    # Growing the chunk count must not delete any valid chunk.
    f = _file()
    vs = MemoryVectorSource(name="m")
    emb = DummyEmbeddingModel()
    big = DocumentPipeline(chunk_size=400, overlap=0, embedding_model=emb, vector_source=vs, collection="c")
    big.ensure_index()
    n1 = len(big.ingest(str(f)))
    small = DocumentPipeline(chunk_size=60, overlap=0, embedding_model=emb, vector_source=vs, collection="c")
    n2 = len(small.ingest(str(f)))
    assert n2 > n1
    assert _stored(vs, "c") == n2  # all chunks present, nothing wrongly trimmed


def test_identical_reingest_issues_no_delete():
    # Re-ingesting identical content (same chunk count) is a pure upsert — no deletes.
    f = _file()
    vs = MagicMock()
    p = DocumentPipeline(chunk_size=60, overlap=0, embedding_model=DummyEmbeddingModel(dim=8), vector_source=vs)
    p.ingest(str(f))
    p.ingest(str(f))
    vs.delete.assert_not_called()
