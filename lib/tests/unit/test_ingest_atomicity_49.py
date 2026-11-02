"""Regression tests for issue #49.

The DocumentRecord was written to the RecordStore *before* the chunk upsert loop, so a
crash / vector-store error mid-loop left a findable document record pointing at a
partially-populated vector store, with no chunk_count to detect the inconsistency.

The record is now written only after every chunk upsert succeeds, and carries
chunk_count. A partial failure therefore leaves no DocumentRecord — re-ingesting from
the source is a clean upsert. No network: the vector source is a stub.
"""
from typing import List

import pytest

from blue_data_doc_source.source.document.pipeline import DocumentPipeline
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel
from blue_data_doc_source.source.document.schema.base import DocumentClassifier


class _StubClassifier(DocumentClassifier):
    """Triggers the schema-record path without an LLM."""
    def classify(self, text: str):
        return "contract", 1.0


class _RecordingVectorSource:
    """Captures upserts; optionally raises on the Nth one to simulate a mid-loop crash."""
    def __init__(self, fail_on_call: int | None = None):
        self.upserts: List[str] = []
        self._fail_on_call = fail_on_call

    def upsert(self, collection, doc_id, doc):
        if self._fail_on_call is not None and len(self.upserts) + 1 == self._fail_on_call:
            raise RuntimeError("simulated vector-store failure")
        self.upserts.append(doc_id)
        return "OK"

    def create_index(self, *a, **k):
        return "OK"


# A document that chunks into several pieces (small chunk_size, no section headers).
_TEXT = " ".join(f"word{i}" for i in range(60))


def _pipeline(vector_source):
    return DocumentPipeline(
        chunk_size=5,
        overlap=1,
        classifier=_StubClassifier(),
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
    )


def test_partial_failure_leaves_no_document_record():
    vs = _RecordingVectorSource(fail_on_call=2)   # crash during the 2nd chunk upsert
    pipeline = _pipeline(vs)

    with pytest.raises(RuntimeError, match="simulated vector-store failure"):
        pipeline.ingest_text(_TEXT, file_name="doc.txt")

    # Some chunks were written to the vector store...
    assert len(vs.upserts) >= 1
    # ...but no DocumentRecord was committed, so nothing points at the partial state (#49).
    document_id = "doc_" + __import__("hashlib").sha256(_TEXT.encode()).hexdigest()[:16]
    assert pipeline.get_record(document_id) is None


def test_successful_ingest_writes_record_with_chunk_count():
    vs = _RecordingVectorSource()
    pipeline = _pipeline(vs)

    records = pipeline.ingest_text(_TEXT, file_name="doc.txt")
    document_id = records[0]["document_id"]

    record = pipeline.get_record(document_id)
    assert record is not None
    # The record is stamped with the real chunk count a reader can validate against.
    assert record["chunk_count"] == len(vs.upserts)
    assert record["chunk_count"] == len(records)


def test_no_record_store_write_without_classifier_or_extractor():
    # When neither a classifier nor a field extractor is set, no DocumentRecord is
    # written at all — the deferred upsert must not change that.
    vs = _RecordingVectorSource()
    pipeline = DocumentPipeline(
        chunk_size=5, overlap=1,
        embedding_model=DummyEmbeddingModel(dim=8), vector_source=vs,
    )
    records = pipeline.ingest_text(_TEXT, file_name="doc.txt")
    document_id = records[0]["document_id"]
    assert pipeline.get_record(document_id) is None
    assert len(vs.upserts) == len(records)
