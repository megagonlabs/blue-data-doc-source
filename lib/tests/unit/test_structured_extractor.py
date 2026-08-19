"""Tests for StructuredFieldExtractor."""
from __future__ import annotations

from typing import TypedDict
from unittest.mock import MagicMock

import pytest

from blue_data_doc_source import StructuredFieldExtractor
from blue_data_doc_source.source.document.schema.structured_extractor import (
    _field_spec_from_schema,
    _is_pydantic,
    _parse_json,
)

# ---------------------------------------------------------------------------
# Optional Pydantic import
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel

    class _EmploymentContract(BaseModel):
        employee_name: str
        start_date: str
        base_salary: str
        non_compete: bool

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

pydantic_only = pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")


# ---------------------------------------------------------------------------
# TypedDict fixture
# ---------------------------------------------------------------------------

class _LeaseInfo(TypedDict):
    tenant: str
    monthly_rent: float
    lease_term_months: int


# ---------------------------------------------------------------------------
# JSON Schema dict fixture
# ---------------------------------------------------------------------------

_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total_due": {"type": "number"},
        "due_date": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# _is_pydantic
# ---------------------------------------------------------------------------

class TestIsPydantic:
    @pydantic_only
    def test_pydantic_model_detected(self):
        assert _is_pydantic(_EmploymentContract) is True

    def test_typeddict_not_pydantic(self):
        assert _is_pydantic(_LeaseInfo) is False

    def test_dict_not_pydantic(self):
        assert _is_pydantic(_INVOICE_SCHEMA) is False

    def test_plain_class_not_pydantic(self):
        class Plain:
            pass
        assert _is_pydantic(Plain) is False


# ---------------------------------------------------------------------------
# _field_spec_from_schema
# ---------------------------------------------------------------------------

class TestFieldSpecFromSchema:
    @pydantic_only
    def test_pydantic_includes_all_fields(self):
        spec = _field_spec_from_schema(_EmploymentContract)
        assert "employee_name" in spec
        assert "start_date" in spec
        assert "base_salary" in spec
        assert "non_compete" in spec

    @pydantic_only
    def test_pydantic_spec_starts_with_extract(self):
        spec = _field_spec_from_schema(_EmploymentContract)
        assert spec.startswith("Extract these fields:")

    def test_typeddict_includes_all_fields(self):
        spec = _field_spec_from_schema(_LeaseInfo)
        assert "tenant" in spec
        assert "monthly_rent" in spec
        assert "lease_term_months" in spec

    def test_json_schema_includes_all_fields(self):
        spec = _field_spec_from_schema(_INVOICE_SCHEMA)
        assert "vendor" in spec
        assert "total_due" in spec
        assert "due_date" in spec

    def test_json_schema_includes_types(self):
        spec = _field_spec_from_schema(_INVOICE_SCHEMA)
        assert "string" in spec
        assert "number" in spec

    def test_empty_json_schema_returns_fallback(self):
        spec = _field_spec_from_schema({"type": "object", "properties": {}})
        assert "Extract structured fields" in spec


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_valid_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert _parse_json(raw) == {"a": 1}

    def test_strips_bare_fences(self):
        raw = "```\n{\"a\": 1}\n```"
        assert _parse_json(raw) == {"a": 1}

    def test_unparseable_returns_raw_response(self):
        result = _parse_json("not json at all")
        assert "raw_response" in result


# ---------------------------------------------------------------------------
# StructuredFieldExtractor._validate
# ---------------------------------------------------------------------------

class TestValidate:
    @pydantic_only
    def test_pydantic_validates_and_coerces(self):
        extractor = StructuredFieldExtractor(schema=_EmploymentContract)
        data = {
            "employee_name": "Alice",
            "start_date": "2026-01-01",
            "base_salary": "$90,000",
            "non_compete": True,
        }
        result = extractor._validate(data)
        assert result["employee_name"] == "Alice"
        assert result["non_compete"] is True

    @pydantic_only
    def test_pydantic_validation_failure_falls_back_to_raw(self):
        extractor = StructuredFieldExtractor(schema=_EmploymentContract)
        # Missing required fields — Pydantic will raise; should fall back
        result = extractor._validate({"irrelevant_key": "value"})
        assert isinstance(result, dict)

    def test_non_pydantic_returns_data_unchanged(self):
        extractor = StructuredFieldExtractor(schema=_LeaseInfo)
        data = {"tenant": "Bob", "monthly_rent": 1500.0}
        assert extractor._validate(data) is data

    def test_json_schema_returns_data_unchanged(self):
        extractor = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
        data = {"vendor": "Acme", "total_due": 5000}
        assert extractor._validate(data) is data


# ---------------------------------------------------------------------------
# StructuredFieldExtractor.extract — mocked OpenAI client
# ---------------------------------------------------------------------------

def _make_mock_client(response_json: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_json
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


class TestExtract:
    def _extractor_with_mock(self, schema, response_json: str) -> StructuredFieldExtractor:
        extractor = StructuredFieldExtractor(schema=schema)
        extractor._client = _make_mock_client(response_json)
        return extractor

    def test_extract_typeddict_returns_parsed_fields(self):
        extractor = self._extractor_with_mock(
            _LeaseInfo,
            '{"tenant": "Alice", "monthly_rent": 1200.0, "lease_term_months": 12}',
        )
        result = extractor.extract("Lease agreement text.", "unknown")
        assert result["tenant"] == "Alice"
        assert result["monthly_rent"] == 1200.0

    def test_extract_json_schema_returns_parsed_fields(self):
        extractor = self._extractor_with_mock(
            _INVOICE_SCHEMA,
            '{"vendor": "Acme Corp", "total_due": 3500, "due_date": "2026-09-01"}',
        )
        result = extractor.extract("Invoice document text.", "invoice")
        assert result["vendor"] == "Acme Corp"
        assert result["total_due"] == 3500

    def test_field_spec_included_in_prompt(self):
        extractor = self._extractor_with_mock(_INVOICE_SCHEMA, '{}')
        extractor.extract("Some text.", "unknown")
        call_args = extractor._client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "vendor" in prompt
        assert "total_due" in prompt

    def test_document_type_ignored(self):
        extractor = self._extractor_with_mock(_LeaseInfo, '{"tenant": "Bob"}')
        # document_type is irrelevant for StructuredFieldExtractor
        result1 = extractor.extract("text", "contract")
        result2 = extractor.extract("text", "unknown")
        assert result1 == result2

    def test_sample_chars_truncates_text(self):
        extractor = StructuredFieldExtractor(schema=_INVOICE_SCHEMA, sample_chars=10)
        extractor._client = _make_mock_client('{}')
        extractor.extract("A" * 1000, "unknown")
        call_args = extractor._client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "A" * 11 not in prompt

    def test_malformed_llm_response_returns_raw_response_key(self):
        extractor = self._extractor_with_mock(_LeaseInfo, "Sorry, I cannot extract that.")
        result = extractor.extract("text", "unknown")
        assert "raw_response" in result

    @pydantic_only
    def test_extract_pydantic_validates_output(self):
        extractor = self._extractor_with_mock(
            _EmploymentContract,
            '{"employee_name": "Carol", "start_date": "2026-03-01", '
            '"base_salary": "$75,000", "non_compete": false}',
        )
        result = extractor.extract("Employment agreement text.", "unknown")
        assert result["employee_name"] == "Carol"
        assert result["non_compete"] is False


# ---------------------------------------------------------------------------
# Integration: plugged into DocumentPipeline via ingest_text
# ---------------------------------------------------------------------------

class TestStructuredExtractorInPipeline:
    def test_pipeline_stores_record_with_structured_extractor(self):
        from blue_data_doc_source import DocumentPipeline

        extractor = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
        extractor._client = _make_mock_client(
            '{"vendor": "Blue Corp", "total_due": 9500, "due_date": "2026-10-01"}'
        )
        pipeline = DocumentPipeline(chunk_size=100, overlap=10, field_extractor=extractor)
        pipeline.ingest_text("Invoice content here.", file_name="inv.txt")

        records = pipeline.find_records()
        assert len(records) == 1
        assert records[0]["document_type"] == "unknown"
        assert records[0]["fields"]["vendor"] == "Blue Corp"

    def test_pipeline_chunks_carry_unknown_type(self):
        from blue_data_doc_source import DocumentPipeline

        extractor = StructuredFieldExtractor(schema=_LeaseInfo)
        extractor._client = _make_mock_client('{"tenant": "Dave"}')
        pipeline = DocumentPipeline(chunk_size=100, overlap=10, field_extractor=extractor)
        chunks = pipeline.ingest_text("Lease content here.", file_name="lease.txt")

        assert all(c.get("document_type") == "unknown" for c in chunks)
