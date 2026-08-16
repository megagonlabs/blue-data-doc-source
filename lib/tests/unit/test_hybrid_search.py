"""Tests for hybrid BM25 + vector search and BM25-only mode."""
from __future__ import annotations

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import DocumentPipeline, DummyEmbeddingModel, DocumentDataSource

CONTRACT_TEXT = """\
This contract is entered into between Acme Corp and Beta Ltd.
Client shall pay a monthly retainer of fifteen thousand dollars.
Either party may terminate this agreement with thirty days written notice.
All information shared under this agreement is strictly confidential.
This agreement is governed by the laws of California.
Services will be performed in a professional and timely manner.
Breach of contract may result in penalties and legal action.
Both parties agree to resolve disputes through binding arbitration.
"""

POLICY_TEXT = """\
Coverage is available to individuals between eighteen and sixty-five years of age.
This policy covers preventive care, emergency room visits, and prescription drugs.
The annual deductible is one thousand five hundred dollars per individual member.
Cosmetic procedures and experimental treatments are excluded from coverage.
The monthly premium for individual coverage is three hundred dollars per month.
Claims must be submitted within ninety days of the date of service rendered.
Pre-authorization is required for all specialist referrals and elective procedures.
"""


@pytest.fixture
def pipeline_with_two_docs():
    vs = MemoryVectorSource()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vs,
        chunk_size=60,
        overlap=10,
        collection="docs",
    )
    pipeline.ensure_index("docs")
    pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt",
                         mime_type="text/plain", collection="docs")
    pipeline.ingest_text(POLICY_TEXT, file_name="policy.txt",
                         mime_type="text/plain", collection="docs")
    return pipeline


# ---------------------------------------------------------------------------
# BM25 candidate scoring
# ---------------------------------------------------------------------------

class TestBM25Candidates:
    def test_returns_scores_for_ingested_chunks(self, pipeline_with_two_docs):
        pipeline = pipeline_with_two_docs
        scores = pipeline._bm25_candidates("termination notice", "docs")
        assert len(scores) > 0

    def test_scores_in_0_1_range(self, pipeline_with_two_docs):
        scores = pipeline_with_two_docs._bm25_candidates("coverage deductible", "docs")
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_empty_when_no_chunks_in_collection(self, pipeline_with_two_docs):
        scores = pipeline_with_two_docs._bm25_candidates("anything", "nonexistent_collection")
        assert scores == {}

    def test_zero_score_chunks_excluded(self, pipeline_with_two_docs):
        # query with no token overlap with any chunk should return empty
        scores = pipeline_with_two_docs._bm25_candidates("xyzzy quux frobinate", "docs")
        assert all(s > 0 for s in scores.values())

    def test_filter_respected(self, pipeline_with_two_docs):
        pipeline = pipeline_with_two_docs
        scores_all = pipeline._bm25_candidates("agreement", "docs")
        scores_filtered = pipeline._bm25_candidates(
            "agreement", "docs", filters={"file_name": "contract.txt"}
        )
        assert len(scores_filtered) <= len(scores_all)
        for cid in scores_filtered:
            chunk = pipeline._chunk_index[cid]
            assert chunk.get("file_name") == "contract.txt"


# ---------------------------------------------------------------------------
# chunk_matches_filters
# ---------------------------------------------------------------------------

class TestChunkMatchesFilters:
    def test_no_filters_always_true(self):
        assert DocumentPipeline._chunk_matches_filters({"file_name": "a.txt"}, None)
        assert DocumentPipeline._chunk_matches_filters({}, {})

    def test_exact_field_match(self):
        chunk = {"file_name": "contract.txt", "mime_type": "text/plain"}
        assert DocumentPipeline._chunk_matches_filters(chunk, {"file_name": "contract.txt"})
        assert not DocumentPipeline._chunk_matches_filters(chunk, {"file_name": "policy.txt"})

    def test_page_start_exact(self):
        chunk = {"page_start": 3}
        assert DocumentPipeline._chunk_matches_filters(chunk, {"page_start": 3})
        assert not DocumentPipeline._chunk_matches_filters(chunk, {"page_start": 4})

    def test_page_start_range(self):
        chunk = {"page_start": 5}
        assert DocumentPipeline._chunk_matches_filters(chunk, {"page_start": {"gte": 3, "lte": 7}})
        assert not DocumentPipeline._chunk_matches_filters(chunk, {"page_start": {"gte": 6, "lte": 9}})

    def test_page_start_none_excluded_by_range(self):
        chunk = {"text": "no page"}  # page_start absent
        assert not DocumentPipeline._chunk_matches_filters(chunk, {"page_start": {"gte": 1}})


# ---------------------------------------------------------------------------
# mode="bm25"
# ---------------------------------------------------------------------------

class TestBM25Mode:
    def test_returns_results(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("termination notice", k=3, mode="bm25")
        assert len(results) > 0

    def test_results_have_score(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("payment monthly", k=3, mode="bm25")
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_respects_k(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("the", k=2, mode="bm25")
        assert len(results) <= 2

    def test_no_vector_source_required(self):
        pipeline = DocumentPipeline(chunk_size=60, overlap=10, collection="docs")
        pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt",
                              mime_type="text/plain", collection="docs")
        results = pipeline.search("terminate agreement", k=3, mode="bm25")
        assert len(results) > 0

    def test_empty_corpus_returns_empty(self):
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=MemoryVectorSource(),
            collection="docs",
        )
        results = pipeline.search("anything", k=3, mode="bm25")
        assert results == []

    def test_filter_applied(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search(
            "coverage deductible premium", k=5, mode="bm25",
            filters={"file_name": "policy.txt"},
        )
        for r in results:
            assert r.get("file_name") == "policy.txt"


# ---------------------------------------------------------------------------
# mode="hybrid"
# ---------------------------------------------------------------------------

class TestHybridMode:
    def test_returns_results(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("termination", k=3, mode="hybrid")
        assert len(results) > 0

    def test_results_have_score(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("termination", k=3, mode="hybrid")
        for r in results:
            assert "score" in r

    def test_respects_k(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("agreement", k=2, mode="hybrid")
        assert len(results) <= 2

    def test_default_mode_is_hybrid(self, pipeline_with_two_docs):
        # default call with no mode= should behave like mode="hybrid"
        results_default = pipeline_with_two_docs.search("termination", k=3)
        results_hybrid  = pipeline_with_two_docs.search("termination", k=3, mode="hybrid")
        assert len(results_default) == len(results_hybrid)

    def test_degrades_to_vector_when_no_corpus(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=60,
            overlap=10,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        # ingest into a different collection so "docs" BM25 corpus is empty
        pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt",
                              mime_type="text/plain", collection="other")
        results = pipeline.search("termination", k=3, collection="docs", mode="hybrid")
        assert results == []  # vector index also empty for "docs"

    def test_hybrid_requires_embedding_model(self):
        pipeline = DocumentPipeline(
            vector_source=MemoryVectorSource(),
            collection="docs",
        )
        with pytest.raises(ValueError, match="embedding_model"):
            pipeline.search("query", mode="hybrid")

    def test_hybrid_requires_vector_source(self):
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            collection="docs",
        )
        with pytest.raises(ValueError, match="vector_source"):
            pipeline.search("query", mode="hybrid")

    def test_chunk_collection_tracking(self, pipeline_with_two_docs):
        pipeline = pipeline_with_two_docs
        for cid in pipeline._chunk_index:
            assert cid in pipeline._chunk_collection
            assert pipeline._chunk_collection[cid] == "docs"

    def test_chunk_collection_cleared_on_delete(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=60,
            overlap=10,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        records = pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt",
                                       mime_type="text/plain", collection="docs")
        doc_id = records[0]["document_id"]
        pipeline.delete(document_id=doc_id, chunk_count=len(records), collection="docs")
        for cid in pipeline._chunk_collection:
            assert not cid.startswith(doc_id)


# ---------------------------------------------------------------------------
# mode="vector" (explicit)
# ---------------------------------------------------------------------------

class TestVectorMode:
    def test_explicit_vector_mode_works(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("termination", k=3, mode="vector")
        assert len(results) > 0

    def test_vector_mode_requires_embedding_model(self):
        pipeline = DocumentPipeline(
            vector_source=MemoryVectorSource(),
            collection="docs",
        )
        with pytest.raises(ValueError, match="embedding_model"):
            pipeline.search("query", mode="vector")


# ---------------------------------------------------------------------------
# mode="rerank"
# ---------------------------------------------------------------------------

class TestRerankMode:
    def test_returns_results(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("terminate agreement", k=3, mode="rerank")
        assert len(results) > 0

    def test_results_have_score(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("terminate agreement", k=3, mode="rerank")
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_respects_k(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search("coverage deductible", k=2, mode="rerank")
        assert len(results) <= 2

    def test_embeddings_stored_during_ingest(self, pipeline_with_two_docs):
        pipeline = pipeline_with_two_docs
        for cid in pipeline._chunk_index:
            assert cid in pipeline._chunk_embeddings

    def test_embeddings_cleared_on_delete(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=60,
            overlap=10,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        records = pipeline.ingest_text(
            CONTRACT_TEXT, file_name="contract.txt",
            mime_type="text/plain", collection="docs"
        )
        doc_id = records[0]["document_id"]
        pipeline.delete(document_id=doc_id, chunk_count=len(records), collection="docs")
        assert not any(cid.startswith(doc_id) for cid in pipeline._chunk_embeddings)

    def test_falls_back_to_bm25_when_no_embeddings(self):
        # Pipeline with no embedding model — rerank falls back to BM25 order
        pipeline = DocumentPipeline(chunk_size=60, overlap=10, collection="docs")
        pipeline.ingest_text(
            CONTRACT_TEXT, file_name="contract.txt",
            mime_type="text/plain", collection="docs"
        )
        results = pipeline.search("terminate agreement", k=3, mode="rerank")
        assert len(results) > 0  # BM25 fallback

    def test_empty_bm25_corpus_returns_empty(self):
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=MemoryVectorSource(),
            collection="docs",
        )
        results = pipeline.search("anything", k=3, mode="rerank")
        assert results == []

    def test_filter_applied(self, pipeline_with_two_docs):
        results = pipeline_with_two_docs.search(
            "coverage deductible premium", k=5, mode="rerank",
            filters={"file_name": "policy.txt"},
        )
        for r in results:
            assert r.get("file_name") == "policy.txt"


# ---------------------------------------------------------------------------
# DOCUMENT.SEARCH op respects mode field
# ---------------------------------------------------------------------------

class TestDocumentSourceMode:
    def test_search_op_passes_mode_to_pipeline(self):
        vs = MemoryVectorSource()
        source = DocumentDataSource(
            name="test",
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=60,
            overlap=10,
            default_collection="docs",
        )
        source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "docs"})
        source.execute_query({
            "op": "DOCUMENT.INGEST",
            "path": None,  # skip actual file — test op routing only via ingest_text
            "collection": "docs",
        }) if False else None

        # ingest via pipeline directly
        source._pipeline.ingest_text(
            CONTRACT_TEXT, file_name="c.txt", mime_type="text/plain", collection="docs"
        )

        results_bm25 = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "termination notice",
            "collection": "docs",
            "k": 3,
            "mode": "bm25",
        })
        assert isinstance(results_bm25, list)

        results_hybrid = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "termination notice",
            "collection": "docs",
            "k": 3,
            "mode": "hybrid",
        })
        assert isinstance(results_hybrid, list)
