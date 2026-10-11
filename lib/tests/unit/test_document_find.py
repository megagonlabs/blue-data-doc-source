"""Tests for DOCUMENT.FIND op and RecordStore."""
from __future__ import annotations

import json
import os
import tempfile

import pytest
from blue_data.source.json.file_json_source import FileJSONSource
from blue_data.source.json.memory_json_source import MemoryJSONSource

from blue_data_doc_source import (
    CoverageReport,
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    FileRecordStore,
    FindResult,
    JsonSourceRecordStore,
    KeywordClassifier,
    MemoryRecordStore,
    RecordStore,
)
from blue_data_doc_source.source.document.schema.schema_store import _matches

# ---------------------------------------------------------------------------
# Sample schema records
# ---------------------------------------------------------------------------

CONTRACT_A = {
    "document_id": "doc_contract_a",
    "document_type": "contract",
    "confidence": 0.78,
    "classifier": "llm",
    "chunk_id": "doc_contract_a_schema",
    "fields": {
        "parties": ["TechCorp Inc.", "BuildRight LLC"],
        "governing_law": "State of Delaware",
        "contract_value": "$180,000",
        "effective_date": "2026-09-01",
        "expiration_date": "2027-08-31",
    },
}

CONTRACT_B = {
    "document_id": "doc_contract_b",
    "document_type": "contract",
    "confidence": 0.65,
    "classifier": "keyword",
    "chunk_id": "doc_contract_b_schema",
    "fields": {
        "parties": ["Alpha Corp", "Beta Inc"],
        "governing_law": "State of California",
        "contract_value": "$45,000",
        "effective_date": "2025-01-01",
        "expiration_date": "2025-12-31",
    },
}

INVOICE = {
    "document_id": "doc_invoice",
    "document_type": "invoice",
    "confidence": 0.80,
    "classifier": "llm",
    "chunk_id": "doc_invoice_schema",
    "fields": {
        "vendor": "Blue Solutions Inc.",
        "bill_to": "Acme Manufacturing Corp",
        "total_due": "$35,154",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-31",
    },
}

POLICY = {
    "document_id": "doc_policy",
    "document_type": "policy",
    "confidence": 0.64,
    "classifier": "llm",
    "chunk_id": "doc_policy_schema",
    "fields": {
        "policy_number": "BSH-2024-004521",
        "insured": "Jane Doe",
        "effective_date": "2024-01-01",
        "deductible": "$1,500",
    },
}

ALL_RECORDS = [CONTRACT_A, CONTRACT_B, INVOICE, POLICY]


def _store_with_all() -> MemoryRecordStore:
    store = MemoryRecordStore()
    for r in ALL_RECORDS:
        store.upsert(r)
    return store


# ---------------------------------------------------------------------------
# MemoryRecordStore basics
# ---------------------------------------------------------------------------

class TestMemoryRecordStore:
    def test_upsert_and_get(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)
        assert store.get("doc_contract_a") == CONTRACT_A

    def test_get_missing_returns_none(self):
        store = MemoryRecordStore()
        assert store.get("nonexistent") is None

    def test_upsert_overwrites(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)
        updated = dict(CONTRACT_A, confidence=0.99)
        store.upsert(updated)
        assert store.get("doc_contract_a")["confidence"] == 0.99

    def test_delete_removes_record(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)
        store.delete("doc_contract_a")
        assert store.get("doc_contract_a") is None

    def test_delete_missing_is_noop(self):
        store = MemoryRecordStore()
        store.delete("nonexistent")  # must not raise

    def test_find_no_filter_returns_all(self):
        store = _store_with_all()
        assert len(store.find()) == 4

    def test_find_empty_filter_returns_all(self):
        store = _store_with_all()
        assert len(store.find({})) == 4


# ---------------------------------------------------------------------------
# FileRecordStore persistence
# ---------------------------------------------------------------------------

class TestFileRecordStore:
    def test_survives_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)  # remove empty file so FileJSONSource initialises it
        try:
            store = FileRecordStore(path)
            store.upsert(CONTRACT_A)
            # reload from disk
            store2 = FileRecordStore(path)
            assert store2.get("doc_contract_a")["document_type"] == "contract"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_persists(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            store = FileRecordStore(path)
            store.upsert(CONTRACT_A)
            store.delete("doc_contract_a")
            store2 = FileRecordStore(path)
            assert store2.get("doc_contract_a") is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_missing_file_starts_empty(self):
        path = tempfile.mktemp(suffix=".json")
        store = FileRecordStore(path)
        assert store.find() == []

    def test_uses_given_path_not_shared_default(self):
        # Regression: FileRecordStore(path) must persist to `path`, not to a shared
        # ./registry.json in the CWD (FileJSONSource's first positional arg is `name`,
        # not the path). Two stores at different paths must be independent.
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fa:
            path_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fb:
            path_b = fb.name
        os.unlink(path_a)
        os.unlink(path_b)
        try:
            store_a = FileRecordStore(path_a)
            store_a.upsert(CONTRACT_A)
            assert os.path.exists(path_a)          # wrote to the given path
            store_b = FileRecordStore(path_b)       # different path → independent
            assert store_b.find() == []
            assert store_b.get("doc_contract_a") is None
        finally:
            for p in (path_a, path_b):
                if os.path.exists(p):
                    os.unlink(p)

    def test_find_works_after_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            store = FileRecordStore(path)
            for r in ALL_RECORDS:
                store.upsert(r)
            store2 = FileRecordStore(path)
            results = store2.find({"document_type": "contract"})
            assert len(results) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Filter: exact match
# ---------------------------------------------------------------------------

class TestFilterExactMatch:
    def test_exact_document_type(self):
        store = _store_with_all()
        results = store.find({"document_type": "contract"})
        assert len(results) == 2
        assert all(r["document_type"] == "contract" for r in results)

    def test_exact_document_type_case_insensitive(self):
        store = _store_with_all()
        results = store.find({"document_type": "CONTRACT"})
        assert len(results) == 2

    def test_exact_nested_field(self):
        store = _store_with_all()
        results = store.find({"fields.governing_law": "State of Delaware"})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_a"

    def test_exact_nested_field_case_insensitive(self):
        store = _store_with_all()
        results = store.find({"fields.governing_law": "state of delaware"})
        assert len(results) == 1

    def test_no_match_returns_empty(self):
        store = _store_with_all()
        results = store.find({"document_type": "financial_report"})
        assert results == []

    def test_multiple_conditions_anded(self):
        store = _store_with_all()
        results = store.find({
            "document_type": "contract",
            "fields.governing_law": "State of California",
        })
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_b"


# ---------------------------------------------------------------------------
# Filter: contains operator
# ---------------------------------------------------------------------------

class TestFilterContains:
    def test_contains_substring(self):
        store = _store_with_all()
        results = store.find({"fields.vendor": {"contains": "Blue Solutions"}})
        assert len(results) == 1
        assert results[0]["document_type"] == "invoice"

    def test_contains_case_insensitive(self):
        store = _store_with_all()
        results = store.find({"fields.vendor": {"contains": "blue solutions"}})
        assert len(results) == 1

    def test_contains_no_match(self):
        store = _store_with_all()
        results = store.find({"fields.vendor": {"contains": "nonexistent"}})
        assert results == []


# ---------------------------------------------------------------------------
# Filter: numeric comparison operators
# ---------------------------------------------------------------------------

class TestFilterNumericComparison:
    def test_gt_strips_currency_symbol(self):
        store = _store_with_all()
        # total_due = "$35,154" → 35154; contract_value = "$180,000" / "$45,000"
        results = store.find({"fields.total_due": {"gt": 30000}})
        assert len(results) == 1
        assert results[0]["document_type"] == "invoice"

    def test_lt_numeric(self):
        store = _store_with_all()
        results = store.find({"fields.contract_value": {"lt": 50000}})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_b"

    def test_gte_numeric(self):
        store = _store_with_all()
        results = store.find({"fields.contract_value": {"gte": 180000}})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_a"

    def test_lte_numeric(self):
        store = _store_with_all()
        results = store.find({"fields.contract_value": {"lte": 45000}})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_b"

    def test_date_string_comparison(self):
        store = _store_with_all()
        # ISO dates compare lexicographically — 2025 < 2026
        results = store.find({"fields.effective_date": {"lt": "2026-01-01"}})
        # contract_b (2025-01-01) and policy (2024-01-01)
        ids = {r["document_id"] for r in results}
        assert "doc_contract_b" in ids
        assert "doc_policy" in ids
        assert "doc_contract_a" not in ids


# ---------------------------------------------------------------------------
# Filter: missing field returns no match
# ---------------------------------------------------------------------------

class TestFilterMissingField:
    def test_missing_nested_field_excluded(self):
        store = _store_with_all()
        # policy has no 'vendor' field
        results = store.find({"fields.vendor": "Blue Solutions Inc."})
        assert all(r["document_type"] != "policy" for r in results)

    def test_missing_top_level_field_excluded(self):
        store = _store_with_all()
        results = store.find({"nonexistent_key": "value"})
        assert results == []


# ---------------------------------------------------------------------------
# DocumentPipeline.find_records
# ---------------------------------------------------------------------------

class TestPipelineFindSchemas:
    def _pipeline_with_classifier(self):
        return DocumentPipeline(
            chunk_size=100,
            overlap=10,
            classifier=KeywordClassifier(),
        )

    def test_find_records_returns_ingested_types(self):
        pipeline = self._pipeline_with_classifier()
        pipeline.ingest_text(
            "This agreement is between parties. Termination requires 30 days notice. "
            "Governing law is California. Indemnification clause applies. Warranties provided. "
            "Arbitration for disputes. Intellectual property owned by client. Liability capped.",
            file_name="c.txt",
        )
        results = pipeline.find_records({"document_type": "contract"})
        assert len(results) == 1

    def test_find_records_no_filter_returns_all(self):
        pipeline = self._pipeline_with_classifier()
        pipeline.ingest_text(
            "Invoice #001. Bill to: Acme. Due date: 2026-08-31. Amount due: $1000. "
            "Subtotal: $900. Tax: $100. Vendor: Supplier Co. Remit to: PO Box 1. "
            "Purchase order: PO-123. Invoice number: INV-001. Unit price: $100.",
            file_name="inv.txt",
        )
        pipeline.ingest_text(
            "This agreement is between parties. Termination clause included. "
            "Governing law: Delaware. Indemnification applies. Warranties included. "
            "Arbitration required. Force majeure included. Intellectual property clause.",
            file_name="c.txt",
        )
        results = pipeline.find_records()
        assert len(results) == 2

    def test_find_records_empty_when_no_classifier(self):
        pipeline = DocumentPipeline(chunk_size=100, overlap=10)
        pipeline.ingest_text("Some text.", file_name="t.txt")
        assert pipeline.find_records() == []


# ---------------------------------------------------------------------------
# DocumentDataSource DOCUMENT.FIND op
# ---------------------------------------------------------------------------

class TestDocumentFindOp:
    def _source(self):
        return DocumentDataSource(
            name="test",
            embedding_model=DummyEmbeddingModel(dim=8),
            chunk_size=100,
            overlap=10,
            classifier=KeywordClassifier(),
        )

    def test_find_op_returns_list(self):
        source = self._source()
        source._pipeline.ingest_text(
            "This agreement is between parties. Termination clause. Governing law: California. "
            "Indemnification applies. Arbitration required. Warranties given. Liability capped. "
            "Intellectual property: work for hire. Force majeure clause included.",
            file_name="c.txt",
        )
        results = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "contract"},
        })
        assert isinstance(results, list)
        assert len(results) == 1

    def test_find_op_no_filter_returns_all(self):
        source = self._source()
        source._pipeline.ingest_text(
            "Invoice #001. Bill to: Acme. Amount due: $500. Due date: 2026-09-01. "
            "Vendor: ABC Inc. Subtotal: $450. Tax: $50. Remit to supplier. "
            "Purchase order PO-99. Invoice number INV-99. Unit price $450.",
            file_name="i.txt",
        )
        results = source.execute_query({"op": "DOCUMENT.FIND"})
        assert isinstance(results, list)

    def test_find_op_no_classifier_returns_empty(self):
        source = DocumentDataSource(name="test", chunk_size=100, overlap=10)
        source._pipeline.ingest_text("Some text.", file_name="t.txt")
        results = source.execute_query({"op": "DOCUMENT.FIND"})
        assert results == []

    def test_record_store_injected_into_source(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)
        source = DocumentDataSource(
            name="test",
            chunk_size=100,
            overlap=10,
            record_store=store,
        )
        results = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "contract"},
        })
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_a"


# ---------------------------------------------------------------------------
# JsonSourceRecordStore — MemoryJSONSource backend
# ---------------------------------------------------------------------------

class TestJsonSourceRecordStoreMemory:
    def _store(self):
        return JsonSourceRecordStore(MemoryJSONSource())

    def test_upsert_and_get(self):
        store = self._store()
        store.upsert(CONTRACT_A)
        result = store.get("doc_contract_a")
        assert result is not None
        assert result["document_type"] == "contract"

    def test_get_missing_returns_none(self):
        store = self._store()
        assert store.get("does_not_exist") is None

    def test_upsert_overwrites(self):
        store = self._store()
        store.upsert(CONTRACT_A)
        updated = {**CONTRACT_A, "confidence": 0.99}
        store.upsert(updated)
        assert store.get("doc_contract_a")["confidence"] == 0.99

    def test_delete_removes_record(self):
        store = self._store()
        store.upsert(CONTRACT_A)
        store.delete("doc_contract_a")
        assert store.get("doc_contract_a") is None

    def test_delete_missing_is_safe(self):
        store = self._store()
        store.delete("does_not_exist")  # must not raise

    def test_find_no_filter_returns_all(self):
        store = self._store()
        for rec in [CONTRACT_A, CONTRACT_B, INVOICE, POLICY]:
            store.upsert(rec)
        assert len(store.find()) == 4

    def test_find_by_document_type(self):
        store = self._store()
        for rec in [CONTRACT_A, CONTRACT_B, INVOICE]:
            store.upsert(rec)
        results = store.find({"document_type": "contract"})
        assert len(results) == 2
        assert all(r["document_type"] == "contract" for r in results)

    def test_find_with_nested_field_filter(self):
        store = self._store()
        for rec in [CONTRACT_A, CONTRACT_B]:
            store.upsert(rec)
        results = store.find({"fields.governing_law": {"contains": "California"}})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_b"

    def test_find_with_numeric_operator(self):
        store = self._store()
        for rec in [CONTRACT_A, CONTRACT_B, INVOICE]:
            store.upsert(rec)
        results = store.find({"fields.contract_value": {"gt": 100000}})
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_a"

    def test_find_empty_store_returns_empty(self):
        store = self._store()
        assert store.find({"document_type": "contract"}) == []


# ---------------------------------------------------------------------------
# JsonSourceRecordStore — FileJSONSource backend
# ---------------------------------------------------------------------------

class TestJsonSourceRecordStoreFile:
    def _store(self, path):
        return JsonSourceRecordStore(FileJSONSource(properties={"connection": {"path": path}}))

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "schemas.json")
            store1 = self._store(path)
            store1.upsert(CONTRACT_A)

            store2 = self._store(path)
            result = store2.get("doc_contract_a")
            assert result is not None
            assert result["document_type"] == "contract"

    def test_find_after_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "schemas.json")
            store1 = self._store(path)
            for rec in [CONTRACT_A, CONTRACT_B, INVOICE]:
                store1.upsert(rec)

            store2 = self._store(path)
            results = store2.find({"document_type": "contract"})
            assert len(results) == 2

    def test_delete_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "schemas.json")
            store1 = self._store(path)
            store1.upsert(CONTRACT_A)
            store1.delete("doc_contract_a")

            store2 = self._store(path)
            assert store2.get("doc_contract_a") is None


# ---------------------------------------------------------------------------
# JsonSourceRecordStore — plugged into DocumentDataSource
# ---------------------------------------------------------------------------

class TestJsonSourceRecordStoreIntegration:
    def test_used_as_record_store_in_document_source(self):
        store = JsonSourceRecordStore(MemoryJSONSource())
        store.upsert(CONTRACT_A)
        store.upsert(INVOICE)

        source = DocumentDataSource(
            name="test",
            chunk_size=100,
            overlap=10,
            record_store=store,
        )
        results = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "contract"},
        })
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_contract_a"


# ---------------------------------------------------------------------------
# Coverage reporting — FindResult / CoverageReport
# ---------------------------------------------------------------------------

class TestFindRecordsCoverage:
    """Unit tests for find_records(coverage=True) on DocumentPipeline."""

    def _pipeline_with_records(self, *records) -> DocumentPipeline:
        pipeline = DocumentPipeline(chunk_size=100, overlap=10)
        for r in records:
            pipeline.record_store.upsert(r)
        return pipeline

    # ------------------------------------------------------------------
    # Return-type checks
    # ------------------------------------------------------------------

    def test_returns_list_when_coverage_false(self):
        pipeline = self._pipeline_with_records(CONTRACT_A)
        result = pipeline.find_records({"document_type": "contract"}, coverage=False)
        assert isinstance(result, list)

    def test_returns_find_result_when_coverage_true(self):
        pipeline = self._pipeline_with_records(CONTRACT_A)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        assert isinstance(result, FindResult)

    # ------------------------------------------------------------------
    # status values
    # ------------------------------------------------------------------

    def test_status_no_data_when_store_empty(self):
        pipeline = DocumentPipeline(chunk_size=100, overlap=10)
        result = pipeline.find_records({"fields.governing_law": "Delaware"}, coverage=True)
        assert result.status == "no_data"
        assert result.coverage.documents_considered == 0
        assert result.coverage.field_coverage_rate == 0.0

    def test_status_ok_when_all_docs_have_field(self):
        # Both contracts have governing_law
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B)
        result = pipeline.find_records({"fields.governing_law": "Delaware"}, coverage=True)
        assert result.status == "ok"
        assert result.coverage.documents_missing_field == 0
        assert result.coverage.field_coverage_rate == 1.0
        assert result.warning is None

    def test_status_partial_when_some_docs_missing_field(self):
        # CONTRACT_A and CONTRACT_B have governing_law; INVOICE does not
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B, INVOICE)
        result = pipeline.find_records({"fields.governing_law": "Delaware"}, coverage=True)
        assert result.status == "partial"
        assert result.coverage.documents_missing_field == 1
        assert result.warning is not None
        assert "governing_law" in result.warning

    # ------------------------------------------------------------------
    # Coverage counts
    # ------------------------------------------------------------------

    def test_coverage_counts_match_store_size(self):
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B, INVOICE, POLICY)
        result = pipeline.find_records({"fields.governing_law": {"contains": "California"}}, coverage=True)
        assert result.coverage.documents_considered == 4
        assert result.coverage.documents_matched == 1  # only CONTRACT_B

    def test_field_coverage_rate_calculation(self):
        # 3 docs total, 1 missing the field → rate = 2/3
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B, INVOICE)
        result = pipeline.find_records({"fields.governing_law": "Delaware"}, coverage=True)
        assert abs(result.coverage.field_coverage_rate - 2 / 3) < 1e-6

    def test_coverage_no_field_keys_means_zero_missing(self):
        # Filtering on document_type only (no fields.*) — missing count should be 0
        pipeline = self._pipeline_with_records(CONTRACT_A, INVOICE)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        assert result.coverage.documents_missing_field == 0
        assert result.status == "ok"

    def test_coverage_none_filter_returns_all_status_ok(self):
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B)
        result = pipeline.find_records(None, coverage=True)
        assert result.status == "ok"
        assert len(result) == 2

    # ------------------------------------------------------------------
    # FindResult iteration compatibility
    # ------------------------------------------------------------------

    def test_find_result_is_iterable(self):
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        ids = [r["document_id"] for r in result]
        assert set(ids) == {"doc_contract_a", "doc_contract_b"}

    def test_find_result_len(self):
        pipeline = self._pipeline_with_records(CONTRACT_A, CONTRACT_B, INVOICE)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        assert len(result) == 2

    def test_find_result_getitem(self):
        pipeline = self._pipeline_with_records(CONTRACT_A)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        assert result[0]["document_id"] == "doc_contract_a"

    def test_find_result_to_dict(self):
        pipeline = self._pipeline_with_records(CONTRACT_A)
        result = pipeline.find_records({"document_type": "contract"}, coverage=True)
        d = result.to_dict()
        assert "records" in d
        assert "status" in d
        assert "coverage" in d
        assert d["coverage"]["documents_considered"] == 1

    # ------------------------------------------------------------------
    # DocumentDataSource DOCUMENT.FIND with coverage=True
    # ------------------------------------------------------------------

    def test_document_source_coverage_true(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)
        store.upsert(INVOICE)

        source = DocumentDataSource(
            name="test",
            chunk_size=100,
            overlap=10,
            record_store=store,
        )
        result = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"fields.governing_law": {"contains": "Delaware"}},
            "coverage": True,
        })
        assert isinstance(result, FindResult)
        assert result.status == "partial"   # INVOICE lacks governing_law
        assert result.coverage.documents_considered == 2
        assert result.coverage.documents_matched == 1

    def test_document_source_coverage_false_returns_list(self):
        store = MemoryRecordStore()
        store.upsert(CONTRACT_A)

        source = DocumentDataSource(
            name="test",
            chunk_size=100,
            overlap=10,
            record_store=store,
        )
        result = source.execute_query({
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "contract"},
        })
        assert isinstance(result, list)
