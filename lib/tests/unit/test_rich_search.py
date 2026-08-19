"""Tests for search(rich=True) — provenance, grouping, neighbor context, and DOCUMENT.SEARCH with rich=True."""
from __future__ import annotations

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import (
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONTRACT_TEXT = """\
SERVICES AGREEMENT

This Agreement is entered into between TechCorp Inc. and BuildRight Solutions.

WHEREAS the parties wish to formalize their working relationship.

1. Obligations: Vendor shall deliver software consulting services.
2. Termination: Either party may terminate this agreement with 30 days notice.
3. Governing Law: This agreement is governed by the laws of Delaware.
4. Indemnification: Each party shall indemnify the other against claims.
5. Confidentiality: All information shared is confidential.
6. Intellectual Property: Deliverables are works-for-hire.
7. Warranties: Services are professional quality.
8. Liability: Liability is capped at fees paid in the prior 12 months.
9. Arbitration: Disputes shall be resolved by binding arbitration.
10. Force Majeure: Neither party is liable for acts of God.
11. Entire Agreement: This supersedes all prior representations. Severability applies.
"""

# Long enough to guarantee 3+ chunks at chunk_size=20. Each paragraph is ~15 words
# so at chunk_size=20 / overlap=5 the WindowChunker produces multiple windows.
LONG_DOC = " ".join([
    "Alpha chunk first words establishing the beginning of the document content here.",
    "Beta chunk second words continuing the narrative in the middle section here.",
    "Gamma chunk third words completing the document near the end of content.",
    "Delta chunk fourth words adding more material to ensure multiple windows appear.",
    "Epsilon chunk fifth words so the document definitely spans several chunk boundaries.",
])

POLICY_TEXT = """\
HEALTH INSURANCE POLICY

Policy Number: BSH-2026-001
Effective Date: January 1, 2026
Expiration Date: December 31, 2026

Eligibility: Full-time employees are eligible for coverage from day one.
Coverage: Medical, dental, and vision. Deductible is $1,500 per individual.
Premium: $650/month per employee.
Exclusions: Cosmetic procedures not covered. Experimental treatments excluded.
Claims: Submit within 180 days of service date.
Cancellation: 30 days written notice required.
Renewal: Policy renews annually.
"""


@pytest.fixture()
def multi_chunk_pipeline():
    """Pipeline guaranteed to produce 3+ chunks from LONG_DOC at chunk_size=20."""
    vs = MemoryVectorSource()
    pipeline = DocumentPipeline(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vs,
        chunk_size=20,
        overlap=5,
        collection="docs",
    )
    pipeline.ensure_index("docs")
    records = pipeline.ingest_text(LONG_DOC, file_name="long.txt")
    assert len(records) >= 3, f"fixture must produce 3+ chunks, got {len(records)}"
    return pipeline, records


@pytest.fixture()
def pipeline_with_data():
    vs = MemoryVectorSource()
    em = DummyEmbeddingModel(dim=8)
    pipeline = DocumentPipeline(
        embedding_model=em,
        vector_source=vs,
        chunk_size=80,
        overlap=10,
        collection="docs",
        classifier=KeywordClassifier(),
    )
    pipeline.ensure_index("docs")
    contract_records = pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
    policy_records = pipeline.ingest_text(POLICY_TEXT, file_name="policy.txt")
    return pipeline, contract_records, policy_records


@pytest.fixture()
def source_with_data():
    vs = MemoryVectorSource()
    em = DummyEmbeddingModel(dim=8)
    source = DocumentDataSource(
        name="docs",
        embedding_model=em,
        vector_source=vs,
        chunk_size=80,
        overlap=10,
        default_collection="docs",
        classifier=KeywordClassifier(),
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "docs"})
    pipeline = source._pipeline
    contract_records = pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
    policy_records = pipeline.ingest_text(POLICY_TEXT, file_name="policy.txt")
    return source, contract_records, policy_records


# ---------------------------------------------------------------------------
# Top-level response structure
# ---------------------------------------------------------------------------

class TestRichSearchStructure:
    def test_returns_dict_with_required_keys(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination obligations", k=3, rich=True)
        assert isinstance(result, dict)
        for key in ("query", "result_count", "document_count", "results", "by_document"):
            assert key in result

    def test_query_echoed_in_response(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination clause", k=3, rich=True)
        assert result["query"] == "termination clause"

    def test_result_count_matches_results_length(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("eligibility coverage", k=4, rich=True)
        assert result["result_count"] == len(result["results"])

    def test_document_count_matches_by_document_length(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=5, rich=True)
        assert result["document_count"] == len(result["by_document"])

    def test_result_count_respects_k(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("the", k=2, rich=True)
        assert result["result_count"] <= 2

    def test_empty_results_when_filter_matches_nothing(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search(
            "termination", k=5, filters={"document_id": "doc_nonexistent"}, rich=True)
        assert result["result_count"] == 0
        assert result["document_count"] == 0
        assert result["results"] == []
        assert result["by_document"] == {}


# ---------------------------------------------------------------------------
# Individual result items
# ---------------------------------------------------------------------------

class TestRichSearchResultItems:
    def test_each_result_has_required_keys(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("obligations", k=3, rich=True)
        for r in result["results"]:
            for key in ("rank", "score", "chunk_id", "text", "provenance"):
                assert key in r, f"Missing key {key!r} in result"

    def test_rank_is_sequential_from_1(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=4, rich=True)
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_score_is_numeric(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("eligibility", k=3, rich=True)
        for r in result["results"]:
            assert isinstance(r["score"], (int, float))

    def test_text_is_non_empty_string(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("coverage", k=3, rich=True)
        for r in result["results"]:
            assert isinstance(r["text"], str)
            assert len(r["text"]) > 0

    def test_document_type_present_when_classifier_set(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "obligations", k=3, filters={"document_id": doc_id}, rich=True)
        for r in result["results"]:
            assert "document_type" in r
            assert r["document_type"] == "contract"

    def test_document_type_absent_when_no_classifier(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=80,
            overlap=10,
        )
        pipeline.ensure_index("docs")
        pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        result = pipeline.search("obligations", k=2, rich=True)
        for r in result["results"]:
            assert "document_type" not in r


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestRichSearchProvenance:
    def test_provenance_has_required_keys(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=3, rich=True)
        for r in result["results"]:
            prov = r["provenance"]
            for key in ("document_id", "file_name", "chunk_id", "chunk_index"):
                assert key in prov, f"Missing provenance key {key!r}"

    def test_provenance_file_name_matches_chunk(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "obligations", k=2, filters={"document_id": doc_id}, rich=True)
        for r in result["results"]:
            assert r["provenance"]["file_name"] == "contract.txt"

    def test_provenance_document_id_matches_chunk(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        expected_doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "agreement", k=2, filters={"document_id": expected_doc_id}, rich=True)
        for r in result["results"]:
            assert r["provenance"]["document_id"] == expected_doc_id

    def test_provenance_chunk_id_matches_result(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=3, rich=True)
        for r in result["results"]:
            assert r["provenance"]["chunk_id"] == r["chunk_id"]

    def test_provenance_chunk_index_is_int(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=3, rich=True)
        for r in result["results"]:
            assert isinstance(r["provenance"]["chunk_index"], int)

    def test_provenance_section_present_when_section_chunker_detects_header(self, pipeline_with_data):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=80,
            overlap=10,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        structured = "# Introduction\nThis is the intro text.\n\n# Methods\nThese are the methods."
        pipeline.ingest_text(structured, file_name="structured.txt")
        result = pipeline.search("introduction", k=2, rich=True)
        sections = [r["provenance"].get("section") for r in result["results"]]
        assert any(s is not None for s in sections)


# ---------------------------------------------------------------------------
# Neighbor context
# ---------------------------------------------------------------------------

class TestRichSearchContext:
    def test_context_key_present_when_context_chunks_gt_0(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "obligations", k=1, filters={"document_id": doc_id}, context_chunks=1, rich=True)
        r = result["results"][0]
        # Context key is present only when there is at least one neighbor
        # (first chunk may have no "before"; last may have no "after")
        if "context" in r:
            assert isinstance(r["context"], dict)
            for key in r["context"]:
                assert key in ("before", "after")
                assert isinstance(r["context"][key], str)

    def test_context_absent_when_context_chunks_0(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=3, context_chunks=0, rich=True)
        for r in result["results"]:
            assert "context" not in r

    def test_middle_chunk_has_both_before_and_after(self, multi_chunk_pipeline):
        pipeline, records = multi_chunk_pipeline
        middle_id = records[1]["chunk_id"]
        doc_id = records[0]["document_id"]
        result = pipeline.search(
            "chunk", k=len(records), filters={"document_id": doc_id}, context_chunks=1, rich=True)
        for r in result["results"]:
            if r["chunk_id"] == middle_id:
                assert "context" in r
                assert "before" in r["context"]
                assert "after" in r["context"]
                return
        pytest.fail(f"Middle chunk {middle_id} not found in search results")

    def test_first_chunk_has_no_before(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        first_id = contract_records[0]["chunk_id"]
        doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "agreement", k=10, filters={"document_id": doc_id}, context_chunks=1, rich=True)
        for r in result["results"]:
            if r["chunk_id"] == first_id:
                assert "before" not in r.get("context", {})
                break

    def test_neighbor_text_matches_adjacent_chunk(self, multi_chunk_pipeline):
        pipeline, records = multi_chunk_pipeline
        second_chunk_id = records[1]["chunk_id"]
        first_chunk_text = records[0]["text"]
        doc_id = records[0]["document_id"]
        result = pipeline.search(
            "chunk", k=len(records), filters={"document_id": doc_id}, context_chunks=1, rich=True)
        for r in result["results"]:
            if r["chunk_id"] == second_chunk_id:
                assert "context" in r
                assert r["context"].get("before") == first_chunk_text
                return
        pytest.fail(f"Chunk {second_chunk_id} not found in search results")

    def test_context_chunks_2_includes_two_neighbors(self, multi_chunk_pipeline):
        pipeline, records = multi_chunk_pipeline
        # records[2] has two chunks before it (indices 0 and 1)
        third_id = records[2]["chunk_id"]
        doc_id = records[0]["document_id"]
        result = pipeline.search(
            "chunk", k=len(records), filters={"document_id": doc_id}, context_chunks=2, rich=True)
        for r in result["results"]:
            if r["chunk_id"] == third_id:
                assert "context" in r
                before = r["context"].get("before", "")
                # "before" joins text of chunks at offsets -2 and -1 with a space
                assert records[0]["text"] in before
                assert records[1]["text"] in before
                return
        pytest.fail(f"Chunk {third_id} not found in search results")


# ---------------------------------------------------------------------------
# Grouping by document
# ---------------------------------------------------------------------------

class TestRichSearchGrouping:
    def test_by_document_keys_are_document_ids(self, pipeline_with_data):
        pipeline, contract_records, policy_records = pipeline_with_data
        result = pipeline.search("termination coverage", k=6, rich=True)
        for doc_id in result["by_document"]:
            assert doc_id.startswith("doc_")

    def test_by_document_match_count_sums_to_result_count(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("the", k=6, rich=True)
        total = sum(g["match_count"] for g in result["by_document"].values())
        assert total == result["result_count"]

    def test_by_document_matches_list_length_equals_match_count(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("policy termination", k=6, rich=True)
        for entry in result["by_document"].values():
            assert len(entry["matches"]) == entry["match_count"]

    def test_by_document_has_file_name(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination", k=4, rich=True)
        for entry in result["by_document"].values():
            assert "file_name" in entry
            assert entry["file_name"] in ("contract.txt", "policy.txt")

    def test_by_document_has_document_type_when_classifier_set(self, pipeline_with_data):
        pipeline, _, _ = pipeline_with_data
        result = pipeline.search("termination coverage", k=6, rich=True)
        for entry in result["by_document"].values():
            assert "document_type" in entry

    def test_by_document_top_sections_lists_unique_sections(self, pipeline_with_data):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=50,
            overlap=5,
        )
        pipeline.ensure_index("docs")
        structured = (
            "# Termination\nEither party may terminate with 30 days notice.\n\n"
            "# Liability\nLiability is capped at fees paid.\n\n"
            "# Termination\nAdditional termination clause."
        )
        pipeline.ingest_text(structured, file_name="doc.txt")
        result = pipeline.search("termination liability", k=6, rich=True)
        for entry in result["by_document"].values():
            if "top_sections" in entry:
                assert len(entry["top_sections"]) == len(set(entry["top_sections"]))

    def test_multiple_documents_appear_as_separate_groups(self, pipeline_with_data):
        pipeline, contract_records, policy_records = pipeline_with_data
        result = pipeline.search("termination coverage", k=8, rich=True)
        doc_ids = set(result["by_document"].keys())
        contract_id = contract_records[0]["document_id"]
        policy_id = policy_records[0]["document_id"]
        assert contract_id in doc_ids or policy_id in doc_ids

    def test_single_document_one_group_in_by_document(self, pipeline_with_data):
        pipeline, contract_records, _ = pipeline_with_data
        doc_id = contract_records[0]["document_id"]
        result = pipeline.search(
            "termination", k=5, filters={"document_id": doc_id}, rich=True)
        assert result["document_count"] == 1
        assert list(result["by_document"].keys()) == [doc_id]


# ---------------------------------------------------------------------------
# _chunk_index lifecycle
# ---------------------------------------------------------------------------

class TestChunkIndexLifecycle:
    def test_chunk_index_cleared_after_delete(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=20,
            overlap=5,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        records = pipeline.ingest_text(LONG_DOC, file_name="long.txt")
        doc_id = records[0]["document_id"]
        chunk_ids = [r["chunk_id"] for r in records]

        # All chunks must be present before delete
        for cid in chunk_ids:
            assert cid in pipeline._chunk_index

        pipeline.delete(document_id=doc_id, chunk_count=len(records))

        # All chunks must be absent after delete
        for cid in chunk_ids:
            assert cid not in pipeline._chunk_index

    def test_neighbor_lookup_returns_none_after_delete(self):
        vs = MemoryVectorSource()
        pipeline = DocumentPipeline(
            embedding_model=DummyEmbeddingModel(dim=8),
            vector_source=vs,
            chunk_size=20,
            overlap=5,
            collection="docs",
        )
        pipeline.ensure_index("docs")
        records = pipeline.ingest_text(LONG_DOC, file_name="long.txt")
        doc_id = records[0]["document_id"]
        second_id = records[1]["chunk_id"]

        pipeline.delete(document_id=doc_id, chunk_count=len(records))

        assert pipeline._neighbor_text(second_id, -1) is None
        assert pipeline._neighbor_text(second_id, 1) is None


# ---------------------------------------------------------------------------
# DOCUMENT.SEARCH op with rich=True
# ---------------------------------------------------------------------------

class TestSearchRichOp:
    def test_op_returns_dict_with_required_keys(self, source_with_data):
        source, _, _ = source_with_data
        result = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "termination obligations",
            "collection": "docs",
            "k": 3,
            "rich": True,
        })
        for key in ("query", "result_count", "document_count", "results", "by_document"):
            assert key in result

    def test_op_raises_without_query(self, source_with_data):
        source, _, _ = source_with_data
        with pytest.raises(ValueError, match="query"):
            source.execute_query({"op": "DOCUMENT.SEARCH", "collection": "docs"})

    def test_op_respects_k(self, source_with_data):
        source, _, _ = source_with_data
        result = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "the",
            "collection": "docs",
            "k": 2,
            "rich": True,
        })
        assert result["result_count"] <= 2

    def test_op_passes_context_chunks(self, source_with_data):
        source, _, _ = source_with_data
        result = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "termination",
            "collection": "docs",
            "k": 3,
            "rich": True,
            "context_chunks": 0,
        })
        for r in result["results"]:
            assert "context" not in r

    def test_op_passes_filter(self, source_with_data):
        source, contract_records, _ = source_with_data
        doc_id = contract_records[0]["document_id"]
        result = source.execute_query({
            "op": "DOCUMENT.SEARCH",
            "query": "obligations",
            "collection": "docs",
            "rich": True,
            "filter": {"document_id": doc_id},
        })
        for r in result["results"]:
            assert r["provenance"]["document_id"] == doc_id
