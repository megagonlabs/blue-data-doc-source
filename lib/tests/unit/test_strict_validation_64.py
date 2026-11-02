"""Regression tests for issue #64.

"Typed/validated extraction" used to silently return the raw, unvalidated dict when
Pydantic validation failed, and never validated TypedDict / JSON Schema output at all.
StructuredFieldExtractor is now strict by default (raises ExtractionValidationError),
validates every schema kind, and in non-strict mode surfaces the outcome under a
reserved "_validation" key. No network — _validate is exercised directly.
"""
import sys

import pytest

from blue_data_doc_source import ExtractionValidationError, StructuredFieldExtractor

_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total_due": {"type": "number"},
    },
    "required": ["vendor", "total_due"],
}


def test_exception_is_importable_and_carries_errors():
    ex = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
    with pytest.raises(ExtractionValidationError) as exc:
        ex._validate({"vendor": "Acme"})          # missing required total_due
    assert isinstance(exc.value, ValueError)
    assert any("total_due" in e for e in exc.value.errors)


def test_non_strict_annotates_validation_status():
    ex = StructuredFieldExtractor(schema=_INVOICE_SCHEMA, strict=False)
    with pytest.warns(UserWarning, match="failed schema validation"):
        result = ex._validate({"vendor": "Acme", "total_due": "free"})
    assert result["_validation"]["status"] == "invalid"
    assert result["_validation"]["errors"]
    # The best-effort fields are still present for inspection.
    assert result["vendor"] == "Acme"


def test_valid_json_schema_is_clean_no_validation_key():
    ex = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
    result = ex._validate({"vendor": "Acme", "total_due": 1200})
    assert "_validation" not in result


def test_json_schema_validation_without_jsonschema_installed(monkeypatch):
    # jsonschema is not a declared dependency — the minimal built-in fallback must still
    # reject malformed values rather than silently accept them.
    monkeypatch.setitem(sys.modules, "jsonschema", None)  # import jsonschema -> ImportError

    strict = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
    with pytest.raises(ExtractionValidationError):
        strict._validate({"vendor": "Acme", "total_due": "not a number"})
    with pytest.raises(ExtractionValidationError, match="total_due"):
        strict._validate({"vendor": "Acme"})  # missing required

    # A valid payload passes the fallback untouched.
    ok = strict._validate({"vendor": "Acme", "total_due": 99})
    assert ok == {"vendor": "Acme", "total_due": 99}


def test_non_object_json_response_is_invalid():
    ex = StructuredFieldExtractor(schema=_INVOICE_SCHEMA)
    with pytest.raises(ExtractionValidationError, match="JSON object"):
        ex._validate([1, 2, 3])  # a JSON array, not an object
