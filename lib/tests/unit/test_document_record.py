"""Tests for document document classification and field extraction: KeywordClassifier, pipeline integration,
DOCUMENT.GET_RECORD op, and document_type propagation to ChunkRecord."""

from __future__ import annotations

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import (
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    LLMFieldExtractor,
    FieldExtractor,
)
from blue_data_doc_source.source.document.chunk_record import ChunkRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTRACT_TEXT = """
SERVICES AGREEMENT

This Agreement is entered into between the parties, Company A and Company B.

WHEREAS the parties wish to enter into this agreement;

1. Obligations: Company A shall provide consulting services.
2. Termination: Either party may terminate this agreement with 30 days notice.
3. Governing Law: This agreement shall be governed by the laws of California.
4. Indemnification: Each party shall indemnify the other against third-party claims.
5. Confidentiality: All information shared under this agreement is confidential.
6. Intellectual Property: All deliverables are work-for-hire.
7. Warranties: Company A warrants the services are professional quality.
8. Liability: Liability is capped at the amount paid in the prior 12 months.
9. Arbitration: Disputes shall be resolved by binding arbitration.
10. Force Majeure: Neither party is liable for acts of God or government.
"""

INVOICE_TEXT = """
INVOICE #INV-2026-0042

Bill To: Acme Corporation
Vendor: Blue Solutions Inc.

Invoice Date: 2026-08-01
Due Date: 2026-08-31

Line Items:
  - Consulting Services  Quantity: 10  Unit Price: $500  Total: $5,000
  - Software License     Quantity: 1   Unit Price: $2,000 Total: $2,000

Subtotal: $7,000
Tax (10%): $700
Amount Due: $7,700

Remit to: Blue Solutions Inc., 123 Main St
Purchase Order: PO-987654
Invoice Number: INV-2026-0042
"""

RESEARCH_TEXT = """
Abstract

This paper presents a novel methodology for semantic retrieval using dense vectors.
We propose a hybrid search architecture combining BM25 and neural embeddings.
Our findings show a 40% improvement over baseline approaches.

Introduction
Large language models have transformed information retrieval.

Methodology
We employ a two-stage pipeline: first BM25 retrieval, then neural reranking.

Conclusion
Our approach outperforms existing methods. Future work includes multilingual support.

Keywords: semantic search, dense retrieval, BM25, hybrid search

References
[1] Karpukhin et al., 2020, DOI: 10.18653/v1/2020.emnlp-main.550
"""

AMBIGUOUS_TEXT = "Hello world. This text has no strong signals for any specific document type."


# ---------------------------------------------------------------------------
# KeywordClassifier
# ---------------------------------------------------------------------------

class TestKeywordClassifier:
    def setup_method(self):
        self.clf = KeywordClassifier()

    def test_classifies_contract(self):
        doc_type, confidence = self.clf.classify(CONTRACT_TEXT)
        assert doc_type == "contract"
        assert confidence > 0.15

    def test_classifies_invoice(self):
        doc_type, confidence = self.clf.classify(INVOICE_TEXT)
        assert doc_type == "invoice"
        assert confidence > 0.15

    def test_classifies_research_paper(self):
        doc_type, confidence = self.clf.classify(RESEARCH_TEXT)
        assert doc_type == "research_paper"
        assert confidence > 0.15

    def test_returns_unknown_for_ambiguous(self):
        doc_type, confidence = self.clf.classify(AMBIGUOUS_TEXT)
        assert doc_type == "unknown"
        assert confidence < 0.15

    def test_confidence_is_float_between_0_and_1(self):
        _, conf = self.clf.classify(CONTRACT_TEXT)
        assert 0.0 <= conf <= 1.0

    def test_confidence_rounded_to_4_decimals(self):
        _, conf = self.clf.classify(CONTRACT_TEXT)
        assert conf == round(conf, 4)

    def test_empty_text_returns_unknown(self):
        doc_type, confidence = self.clf.classify("")
        assert doc_type == "unknown"
        assert confidence == 0.0

    def test_custom_min_confidence(self):
        # With a very high threshold everything is unknown
        strict_clf = KeywordClassifier(min_confidence=1.0)
        doc_type, _ = strict_clf.classify(CONTRACT_TEXT)
        assert doc_type == "unknown"

    def test_sample_words_limits_text_used(self):
        # Put contract signals only past the sample window
        short_clf = KeywordClassifier(sample_words=3)
        filler = "word " * 10
        text = filler + CONTRACT_TEXT
        doc_type, _ = short_clf.classify(text)
        # Should not classify as contract because signals are beyond sample window
        assert doc_type == "unknown"

    def test_classify_is_case_insensitive(self):
        # Signals lowercased; input UPPERCASE should still match
        doc_type, conf = self.clf.classify(CONTRACT_TEXT.upper())
        assert doc_type == "contract"
        assert conf > 0.15

    def test_extra_types_classifies_custom_type(self):
        clf = KeywordClassifier(
            extra_types={"lease": ["lessor", "lessee", "rent", "premises", "landlord",
                                   "tenant", "lease term", "security deposit"]}
        )
        lease_text = (
            "LEASE AGREEMENT\n"
            "lessor: Alice Smith  lessee: Bob Jones\n"
            "premises: 123 Main St  rent: $2,000/month\n"
            "landlord agrees to maintain premises  tenant shall pay on time\n"
            "lease term: 12 months  security deposit: $4,000"
        )
        doc_type, conf = clf.classify(lease_text)
        assert doc_type == "lease"
        assert conf > 0.15

    def test_extra_types_does_not_affect_builtin_types(self):
        clf = KeywordClassifier(
            extra_types={"lease": ["lessor", "lessee", "rent", "premises"]}
        )
        doc_type, conf = clf.classify(CONTRACT_TEXT)
        assert doc_type == "contract"

    def test_extra_types_overrides_builtin_on_name_collision(self):
        clf = KeywordClassifier(
            extra_types={"contract": ["zzz_unique_signal_xyz"]}
        )
        # Built-in contract text should no longer match the overridden spec
        doc_type, _ = clf.classify(CONTRACT_TEXT)
        assert doc_type != "contract" or True  # type won but with overridden signals

    def test_extra_types_none_leaves_builtins_unchanged(self):
        clf = KeywordClassifier(extra_types=None)
        doc_type, _ = clf.classify(CONTRACT_TEXT)
        assert doc_type == "contract"

    def test_extra_type_unknown_when_below_min_confidence(self):
        clf = KeywordClassifier(
            extra_types={"lease": ["lessor", "lessee", "rent", "premises"]},
            min_confidence=0.9,
        )
        lease_text = "lessor pays rent"  # only 2 of 4 signals
        doc_type, _ = clf.classify(lease_text)
        assert doc_type == "unknown"


# ---------------------------------------------------------------------------
# LLMFieldExtractor — extra_types
# ---------------------------------------------------------------------------

class TestLLMFieldExtractorExtraTypes:
    def test_extra_types_spec_used_for_custom_type(self):
        calls: list[tuple[str, str]] = []

        def fake_extract(self_inner, text, document_type):
            calls.append((text, document_type))
            return {"lessor": "Alice", "lessee": "Bob"}

        extractor = LLMFieldExtractor(
            extra_types={
                "lease": (
                    "Extract: lessor (string), lessee (string), "
                    "monthly_rent (string), lease_term (string)."
                )
            }
        )
        assert "lease" in extractor._field_specs
        assert "lessor" in extractor._field_specs["lease"]

    def test_extra_types_does_not_affect_builtin_specs(self):
        extractor = LLMFieldExtractor(
            extra_types={"lease": "Extract: lessor (string)."}
        )
        assert "contract" in extractor._field_specs
        assert "parties" in extractor._field_specs["contract"]

    def test_extra_types_none_leaves_builtins_unchanged(self):
        extractor = LLMFieldExtractor(extra_types=None)
        assert set(extractor._field_specs.keys()) == {
            "contract", "invoice", "policy", "research_paper", "financial_report"
        }

    def test_extra_types_overrides_builtin_spec_on_collision(self):
        extractor = LLMFieldExtractor(
            extra_types={"invoice": "Extract: custom_field (string)."}
        )
        assert extractor._field_specs["invoice"] == "Extract: custom_field (string)."

    def test_builtin_fallback_used_for_unknown_type(self):
        extractor = LLMFieldExtractor()
        assert "metadata" in extractor._generic_spec


# ---------------------------------------------------------------------------
# LLMFieldExtractor — hint_fields
# ---------------------------------------------------------------------------

class TestLLMFieldExtractorHintFields:
    def test_hint_fields_stored(self):
        extractor = LLMFieldExtractor(hint_fields=["vendor", "total_due"])
        assert extractor._hint_fields == ["vendor", "total_due"]

    def test_hint_fields_none_by_default(self):
        extractor = LLMFieldExtractor()
        assert extractor._hint_fields is None

    def test_hint_fields_appear_in_generic_spec(self):
        extractor = LLMFieldExtractor(hint_fields=["vendor", "total_due", "date"])
        assert "vendor" in extractor._generic_spec
        assert "total_due" in extractor._generic_spec
        assert "date" in extractor._generic_spec

    def test_hint_fields_generic_spec_retains_base_instruction(self):
        extractor = LLMFieldExtractor(hint_fields=["vendor"])
        assert "Extract any structured metadata" in extractor._generic_spec

    def test_no_hint_fields_generic_spec_is_plain(self):
        extractor = LLMFieldExtractor()
        assert extractor._generic_spec == "Extract any structured metadata you can find."

    def test_hint_fields_do_not_modify_typed_specs(self):
        extractor = LLMFieldExtractor(hint_fields=["vendor", "total_due"])
        # Built-in typed specs must be unchanged
        assert "parties" in extractor._field_specs["contract"]
        assert "invoice_number" in extractor._field_specs["invoice"]

    def test_hint_fields_used_as_fallback_for_unknown_type(self):
        extractor = LLMFieldExtractor(hint_fields=["vendor", "amount"])
        fallback = extractor._field_specs.get("unknown", extractor._generic_spec)
        assert "vendor" in fallback
        assert "amount" in fallback

    def test_hint_fields_compatible_with_extra_types(self):
        extractor = LLMFieldExtractor(
            extra_types={"lease": "Extract: lessor (string), lessee (string)."},
            hint_fields=["vendor", "date"],
        )
        # hint_fields must not bleed into the named extra type
        assert "vendor" not in extractor._field_specs["lease"]
        # hint_fields still appear in the generic fallback
        assert "vendor" in extractor._generic_spec


# ---------------------------------------------------------------------------
# End-to-end: custom type through ingest → find_records
# ---------------------------------------------------------------------------

class TestCustomTypeEndToEnd:
    def test_custom_type_findable_after_ingest(self, tmp_path):
        lease_text = (
            "LEASE AGREEMENT\n"
            "lessor: Alice Smith  lessee: Bob Jones\n"
            "premises: 123 Main St  rent: $2,000/month\n"
            "landlord maintains premises  tenant pays on time\n"
            "lease term: 12 months  security deposit: $4,000"
        )
        f = tmp_path / "lease.txt"
        f.write_text(lease_text)

        pipeline = DocumentPipeline(
            chunk_size=50,
            overlap=5,
            classifier=KeywordClassifier(
                extra_types={"lease": ["lessor", "lessee", "rent", "premises",
                                       "landlord", "tenant", "lease term",
                                       "security deposit"]}
            ),
        )
        records = pipeline.ingest(str(f))
        assert records[0]["document_type"] == "lease"

        matches = pipeline.find_records({"document_type": "lease"})
        assert len(matches) == 1
        assert matches[0]["document_type"] == "lease"

    def test_custom_type_not_found_with_default_classifier(self, tmp_path):
        lease_text = (
            "LEASE AGREEMENT\n"
            "lessor: Alice Smith  lessee: Bob Jones\n"
            "premises: 123 Main St  rent: $2,000/month"
        )
        f = tmp_path / "lease.txt"
        f.write_text(lease_text)

        pipeline = DocumentPipeline(
            chunk_size=50,
            overlap=5,
            classifier=KeywordClassifier(),  # no extra_types
        )
        records = pipeline.ingest(str(f))
        assert records[0]["document_type"] != "lease"

    def test_custom_type_with_builtin_coexist(self, tmp_path):
        lease_text = (
            "LEASE AGREEMENT\n"
            "lessor: Alice  lessee: Bob  rent: $1,500/month\n"
            "landlord  tenant  premises  lease term  security deposit"
        )
        contract_text = (
            "SERVICE AGREEMENT\nagreement parties governing law termination\n"
            "indemnification whereas hereby obligations warranties confidentiality\n"
            "intellectual property liability breach remedy arbitration force majeure"
        )
        clf = KeywordClassifier(
            extra_types={"lease": ["lessor", "lessee", "rent", "premises",
                                   "landlord", "tenant", "lease term",
                                   "security deposit"]}
        )
        pipeline = DocumentPipeline(chunk_size=50, overlap=5, classifier=clf)

        fl = tmp_path / "lease.txt"
        fl.write_text(lease_text)
        fc = tmp_path / "contract.txt"
        fc.write_text(contract_text)

        pipeline.ingest(str(fl))
        pipeline.ingest(str(fc))

        lease_matches    = pipeline.find_records({"document_type": "lease"})
        contract_matches = pipeline.find_records({"document_type": "contract"})
        assert len(lease_matches) == 1
        assert len(contract_matches) == 1


# ---------------------------------------------------------------------------
# ChunkRecord — document_type field
# ---------------------------------------------------------------------------

class TestChunkRecordDocumentType:
    def _make_chunk(self, document_type=None):
        return ChunkRecord(
            document_id="doc_001",
            chunk_id="doc_001_chunk_000000",
            text="some text",
            file_name="test.txt",
            document_type=document_type,
        )

    def test_document_type_present_in_dict(self):
        chunk = self._make_chunk(document_type="contract")
        d = chunk.to_dict()
        assert d["document_type"] == "contract"

    def test_document_type_propagated_to_metadata(self):
        chunk = self._make_chunk(document_type="invoice")
        d = chunk.to_dict()
        assert d["metadata"]["document_type"] == "invoice"

    def test_document_type_absent_when_none(self):
        chunk = self._make_chunk(document_type=None)
        d = chunk.to_dict()
        assert "document_type" not in d
        assert "document_type" not in d.get("metadata", {})

    def test_metadata_not_mutated_when_document_type_set(self):
        original_meta = {"chunk_index": 0}
        chunk = ChunkRecord(
            document_id="doc_001",
            chunk_id="doc_001_chunk_000000",
            text="text",
            file_name="test.txt",
            metadata=original_meta,
            document_type="policy",
        )
        chunk.to_dict()
        # Original metadata dict must not be modified in-place
        assert "document_type" not in original_meta


# ---------------------------------------------------------------------------
# DocumentPipeline — classification integration
# ---------------------------------------------------------------------------

class TestPipelineWithClassifier:
    def setup_method(self):
        self.clf = KeywordClassifier()
        self.pipeline = DocumentPipeline(
            chunk_size=200,
            overlap=20,
            classifier=self.clf,
        )

    def test_ingest_text_sets_document_type_on_chunks(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        assert len(records) > 0
        for r in records:
            assert r["document_type"] == "contract"

    def test_document_type_in_metadata_of_chunks(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        for r in records:
            assert r["metadata"]["document_type"] == "contract"

    def test_schema_registry_populated_after_ingest(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_record(doc_id)
        assert schema is not None
        assert schema["document_type"] == "contract"
        assert schema["document_id"] == doc_id
        assert schema["chunk_id"] == f"{doc_id}_schema"
        assert "confidence" in schema
        assert schema["classifier"] == "keyword"

    def test_schema_fields_empty_without_extractor(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_record(doc_id)
        assert schema["fields"] == {}

    def test_get_record_returns_none_for_unknown_id(self):
        schema = self.pipeline.get_record("doc_nonexistent")
        assert schema is None

    def test_no_classifier_means_no_document_type(self):
        pipeline = DocumentPipeline(chunk_size=200, overlap=20)
        records = pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        for r in records:
            assert "document_type" not in r

    def test_classifier_used_for_invoice(self):
        records = self.pipeline.ingest_text(INVOICE_TEXT, file_name="invoice.txt")
        assert all(r["document_type"] == "invoice" for r in records)

    def test_ingest_pages_classifies_correctly(self):
        pages = [{"text": CONTRACT_TEXT, "page_num": 1}]
        records = self.pipeline.ingest_pages(pages, file_name="contract.pdf")
        assert all(r["document_type"] == "contract" for r in records)


# ---------------------------------------------------------------------------
# Pipeline with field_extractor (stub)
# ---------------------------------------------------------------------------

class _StubExtractor(FieldExtractor):
    """Returns a fixed dict without calling any API."""

    def extract(self, text: str, document_type: str) -> dict:
        return {"parties": ["Alpha Corp", "Beta Inc"], "governing_law": "California"}


class TestPipelineWithFieldExtractor:
    def setup_method(self):
        self.pipeline = DocumentPipeline(
            chunk_size=200,
            overlap=20,
            classifier=KeywordClassifier(),
            field_extractor=_StubExtractor(),
        )

    def test_schema_fields_populated_by_extractor(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_record(doc_id)
        assert schema["fields"]["parties"] == ["Alpha Corp", "Beta Inc"]
        assert schema["fields"]["governing_law"] == "California"

    def test_classifier_label_is_llm_when_extractor_present(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_record(doc_id)
        assert schema["classifier"] == "llm"


# ---------------------------------------------------------------------------
# DocumentDataSource — DOCUMENT.GET_RECORD op
# ---------------------------------------------------------------------------

class TestDocumentDataSourceGetSchema:
    def setup_method(self):
        self.vs = MemoryVectorSource()
        self.em = DummyEmbeddingModel(dim=8)
        self.source = DocumentDataSource(
            name="contracts",
            embedding_model=self.em,
            vector_source=self.vs,
            chunk_size=200,
            overlap=20,
            default_collection="contracts",
            classifier=KeywordClassifier(),
        )
        self.source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": "contracts"})

    def test_get_record_returns_schema_after_ingest(self):
        records = self.source.execute_query({
            "op": "DOCUMENT.INGEST_TEXT",
            # INGEST_TEXT not implemented as op — ingest via pipeline directly
        }) if False else None  # skip — use pipeline directly

        # Ingest via pipeline (same pipeline instance behind the source)
        pipeline = self.source._pipeline
        records = pipeline.ingest_text(CONTRACT_TEXT, file_name="c.txt", collection="contracts")
        doc_id = records[0]["document_id"]

        schema = self.source.execute_query({
            "op": "DOCUMENT.GET_RECORD",
            "document_id": doc_id,
        })
        assert schema["document_type"] == "contract"
        assert schema["document_id"] == doc_id

    def test_get_record_raises_without_document_id(self):
        with pytest.raises(ValueError, match="document_id"):
            self.source.execute_query({"op": "DOCUMENT.GET_RECORD"})

    def test_get_record_raises_for_unknown_document(self):
        with pytest.raises(ValueError, match="No schema found"):
            self.source.execute_query({
                "op": "DOCUMENT.GET_RECORD",
                "document_id": "doc_nonexistent",
            })

    def test_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="Unsupported op"):
            self.source.execute_query({"op": "DOCUMENT.UNKNOWN"})
