"""Tests for document schema inference: KeywordClassifier, pipeline integration,
DOCUMENT.GET_SCHEMA op, and document_type propagation to ChunkRecord."""

from __future__ import annotations

import pytest

from blue_data.source.vector.memory_vector_source import MemoryVectorSource

from blue_data_doc_source import (
    DocumentDataSource,
    DocumentPipeline,
    DummyEmbeddingModel,
    KeywordClassifier,
    LLMSchemaExtractor,
    SchemaExtractor,
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
        schema = self.pipeline.get_schema(doc_id)
        assert schema is not None
        assert schema["document_type"] == "contract"
        assert schema["document_id"] == doc_id
        assert schema["chunk_id"] == f"{doc_id}_schema"
        assert "confidence" in schema
        assert schema["classifier"] == "keyword"

    def test_schema_fields_empty_without_extractor(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_schema(doc_id)
        assert schema["fields"] == {}

    def test_get_schema_returns_none_for_unknown_id(self):
        schema = self.pipeline.get_schema("doc_nonexistent")
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
# Pipeline with schema_extractor (stub)
# ---------------------------------------------------------------------------

class _StubExtractor(SchemaExtractor):
    """Returns a fixed dict without calling any API."""

    def extract(self, text: str, document_type: str) -> dict:
        return {"parties": ["Alpha Corp", "Beta Inc"], "governing_law": "California"}


class TestPipelineWithSchemaExtractor:
    def setup_method(self):
        self.pipeline = DocumentPipeline(
            chunk_size=200,
            overlap=20,
            classifier=KeywordClassifier(),
            schema_extractor=_StubExtractor(),
        )

    def test_schema_fields_populated_by_extractor(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_schema(doc_id)
        assert schema["fields"]["parties"] == ["Alpha Corp", "Beta Inc"]
        assert schema["fields"]["governing_law"] == "California"

    def test_classifier_label_is_llm_when_extractor_present(self):
        records = self.pipeline.ingest_text(CONTRACT_TEXT, file_name="contract.txt")
        doc_id = records[0]["document_id"]
        schema = self.pipeline.get_schema(doc_id)
        assert schema["classifier"] == "llm"


# ---------------------------------------------------------------------------
# DocumentDataSource — DOCUMENT.GET_SCHEMA op
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

    def test_get_schema_returns_schema_after_ingest(self):
        records = self.source.execute_query({
            "op": "DOCUMENT.INGEST_TEXT",
            # INGEST_TEXT not implemented as op — ingest via pipeline directly
        }) if False else None  # skip — use pipeline directly

        # Ingest via pipeline (same pipeline instance behind the source)
        pipeline = self.source._pipeline
        records = pipeline.ingest_text(CONTRACT_TEXT, file_name="c.txt", collection="contracts")
        doc_id = records[0]["document_id"]

        schema = self.source.execute_query({
            "op": "DOCUMENT.GET_SCHEMA",
            "document_id": doc_id,
        })
        assert schema["document_type"] == "contract"
        assert schema["document_id"] == doc_id

    def test_get_schema_raises_without_document_id(self):
        with pytest.raises(ValueError, match="document_id"):
            self.source.execute_query({"op": "DOCUMENT.GET_SCHEMA"})

    def test_get_schema_raises_for_unknown_document(self):
        with pytest.raises(ValueError, match="No schema found"):
            self.source.execute_query({
                "op": "DOCUMENT.GET_SCHEMA",
                "document_id": "doc_nonexistent",
            })

    def test_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="Unsupported op"):
            self.source.execute_query({"op": "DOCUMENT.UNKNOWN"})
