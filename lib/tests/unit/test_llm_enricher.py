"""Tests for LLMEnricher."""
from __future__ import annotations

import json

import pytest

from blue_data_doc_source.source.document.enrichment import BasicEnricher, LLMEnricher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Acme Corp entered into a service agreement with Beta Ltd on January 15, 2024. "
    "The monthly retainer is $5,000 payable within 30 days of invoice. "
    "Either party may terminate with 60 days written notice."
)

_FULL_RESPONSE = {
    "entities": [
        {"text": "Acme Corp", "type": "ORG"},
        {"text": "Beta Ltd", "type": "ORG"},
        {"text": "January 15, 2024", "type": "DATE"},
        {"text": "$5,000", "type": "MONEY"},
    ],
    "keyphrases": ["service agreement", "monthly retainer", "termination notice"],
    "relationships": [
        {"subject": "Acme Corp", "predicate": "entered into", "object": "service agreement"},
        {"subject": "Either party", "predicate": "may terminate with", "object": "60 days notice"},
    ],
    "semantic_flags": {
        "is_legal": True,
        "is_financial": True,
        "is_policy": False,
        "is_technical": False,
        "is_medical": False,
        "has_obligations": True,
        "has_deadlines": True,
        "has_monetary_values": True,
    },
    "language": "en",
    "summary": "Acme Corp and Beta Ltd agreed to a monthly retainer with termination provisions.",
}


def stub_llm(response_dict: dict):
    """Return an llm_fn that always returns the given dict serialised as JSON."""
    payload = json.dumps(response_dict)
    return lambda _prompt: payload


def raising_llm(exc=RuntimeError("llm error")):
    """Return an llm_fn that always raises."""
    def _fn(_prompt):
        raise exc
    return _fn


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_entities_and_semantic_hints(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        result = enricher.enrich(SAMPLE_TEXT)
        assert "entities" in result
        assert "semantic_hints" in result

    def test_semantic_hints_required_keys(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        hints = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]
        for key in ("word_count", "sentence_count", "keyphrases", "relationships",
                    "semantic_flags", "language", "summary"):
            assert key in hints, f"missing key: {key}"

    def test_entities_list_of_dicts(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        entities = enricher.enrich(SAMPLE_TEXT)["entities"]
        assert isinstance(entities, list)
        for e in entities:
            assert "text" in e and "type" in e


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

class TestFieldExtraction:
    def test_entities_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        entities = enricher.enrich(SAMPLE_TEXT)["entities"]
        types = {e["type"] for e in entities}
        assert "ORG" in types
        assert "DATE" in types
        assert "MONEY" in types

    def test_keyphrases_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        kp = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["keyphrases"]
        assert "monthly retainer" in kp

    def test_relationships_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        rels = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["relationships"]
        assert len(rels) == 2
        assert all("subject" in r and "predicate" in r and "object" in r for r in rels)

    def test_semantic_flags_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        flags = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["semantic_flags"]
        assert flags["is_legal"] is True
        assert flags["is_financial"] is True
        assert flags["is_medical"] is False

    def test_language_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        lang = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["language"]
        assert lang == "en"

    def test_summary_extracted(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        summary = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["summary"]
        assert isinstance(summary, str) and len(summary) > 0


# ---------------------------------------------------------------------------
# word_count / sentence_count are computed locally, not from LLM
# ---------------------------------------------------------------------------

class TestComputedFields:
    def test_word_count_matches_text(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        result = enricher.enrich(SAMPLE_TEXT)
        assert result["semantic_hints"]["word_count"] == len(SAMPLE_TEXT.split())

    def test_word_count_independent_of_llm(self):
        # Even if LLM returns a different word_count, we compute our own
        bad = dict(_FULL_RESPONSE)
        enricher = LLMEnricher(llm_fn=stub_llm(bad))
        result = enricher.enrich("one two three")
        assert result["semantic_hints"]["word_count"] == 3

    def test_sentence_count(self):
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE))
        result = enricher.enrich("Hello. World! How are you?")
        assert result["semantic_hints"]["sentence_count"] == 3


# ---------------------------------------------------------------------------
# JSON parsing edge cases
# ---------------------------------------------------------------------------

class TestJsonParsing:
    def test_strips_markdown_code_fences(self):
        payload = "```json\n" + json.dumps(_FULL_RESPONSE) + "\n```"
        enricher = LLMEnricher(llm_fn=lambda _: payload)
        result = enricher.enrich(SAMPLE_TEXT)
        assert result["semantic_hints"]["language"] == "en"

    def test_strips_plain_code_fences(self):
        payload = "```\n" + json.dumps(_FULL_RESPONSE) + "\n```"
        enricher = LLMEnricher(llm_fn=lambda _: payload)
        result = enricher.enrich(SAMPLE_TEXT)
        assert result["semantic_hints"]["language"] == "en"

    def test_json_embedded_in_prose(self):
        payload = "Here is the result:\n" + json.dumps(_FULL_RESPONSE) + "\nDone."
        enricher = LLMEnricher(llm_fn=lambda _: payload)
        result = enricher.enrich(SAMPLE_TEXT)
        assert result["semantic_hints"]["language"] == "en"

    def test_malformed_entities_filtered(self):
        bad = dict(_FULL_RESPONSE, entities=[
            {"text": "Good", "type": "ORG"},
            {"text": "Missing type"},       # no type
            {"type": "ORG"},                # no text
            "not a dict",                   # wrong shape
        ])
        enricher = LLMEnricher(llm_fn=stub_llm(bad))
        entities = enricher.enrich(SAMPLE_TEXT)["entities"]
        assert all("text" in e and "type" in e for e in entities)
        assert len(entities) == 1


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_on_llm_exception(self):
        fallback = BasicEnricher()
        enricher = LLMEnricher(llm_fn=raising_llm(), fallback=fallback)
        result = enricher.enrich(SAMPLE_TEXT)
        # BasicEnricher returns keywords, not keyphrases
        assert "keywords" in result["semantic_hints"]

    def test_fallback_on_parse_error(self):
        fallback = BasicEnricher()
        enricher = LLMEnricher(llm_fn=lambda _: "not json at all", fallback=fallback)
        result = enricher.enrich(SAMPLE_TEXT)
        assert "keywords" in result["semantic_hints"]

    def test_minimal_when_no_fallback_and_llm_fails(self):
        enricher = LLMEnricher(llm_fn=raising_llm())
        result = enricher.enrich(SAMPLE_TEXT)
        assert result["entities"] == []
        assert result["semantic_hints"]["keyphrases"] == []
        assert result["semantic_hints"]["word_count"] == len(SAMPLE_TEXT.split())

    def test_minimal_preserves_word_count_on_empty_text(self):
        enricher = LLMEnricher(llm_fn=raising_llm())
        result = enricher.enrich("")
        assert result["semantic_hints"]["word_count"] == 0
        assert result["semantic_hints"]["sentence_count"] == 0


# ---------------------------------------------------------------------------
# max_chunk_chars truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_prompt_truncated_to_max_chars(self):
        received: list[str] = []

        def capture(prompt: str) -> str:
            received.append(prompt)
            return json.dumps(_FULL_RESPONSE)

        long_text = "word " * 1000  # 5000 chars
        enricher = LLMEnricher(llm_fn=capture, max_chunk_chars=100)
        enricher.enrich(long_text)
        # The chunk portion of the prompt must not exceed 100 chars
        assert len(received) == 1
        # Confirm the full text was NOT sent (it's 5000 chars)
        assert long_text not in received[0]

    def test_word_count_reflects_full_text_not_truncated(self):
        """word_count is always from the original text, not the truncated version."""
        enricher = LLMEnricher(llm_fn=stub_llm(_FULL_RESPONSE), max_chunk_chars=10)
        text = "one two three four five"
        result = enricher.enrich(text)
        assert result["semantic_hints"]["word_count"] == 5


# ---------------------------------------------------------------------------
# Partial LLM responses (missing optional fields)
# ---------------------------------------------------------------------------

class TestPartialResponse:
    def test_missing_keyphrases_defaults_to_empty_list(self):
        minimal = {"entities": [], "semantic_flags": {}, "language": "en", "summary": ""}
        enricher = LLMEnricher(llm_fn=stub_llm(minimal))
        hints = enricher.enrich(SAMPLE_TEXT)["semantic_hints"]
        assert hints["keyphrases"] == []
        assert hints["relationships"] == []

    def test_missing_language_defaults_to_empty_string(self):
        no_lang = {k: v for k, v in _FULL_RESPONSE.items() if k != "language"}
        enricher = LLMEnricher(llm_fn=stub_llm(no_lang))
        assert enricher.enrich(SAMPLE_TEXT)["semantic_hints"]["language"] == ""
