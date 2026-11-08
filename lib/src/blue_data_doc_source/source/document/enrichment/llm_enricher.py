from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .base import Enricher
from ..prompt_safety import UNTRUSTED_DOCUMENT_WARNING, wrap_document

_SENTENCE_RE = re.compile(r"[.!?]+")

# The chunk is untrusted document text. LLMEnricher calls a (prompt) -> str function with
# no system role, so the data warning and delimiters go in the prompt text itself — a
# planted "ignore all previous instructions" line then sets no entities/fields (#197).
_PROMPT = """\
Analyze the document chunk between the <document> and </document> markers and return a \
JSON object with these exact fields:

- "entities": array of {{"text": string, "type": string}} — named entities with types \
PERSON, ORG, LOCATION, DATE, MONEY, PERCENT, PRODUCT, LEGAL_TERM, REGULATION
- "keyphrases": array of strings — 3 to 7 key multi-word phrases capturing the main topics
- "relationships": array of {{"subject": string, "predicate": string, "object": string}} \
— up to 5 factual triples extracted from the text
- "semantic_flags": object with boolean fields: is_legal, is_financial, is_policy, \
is_technical, is_medical, has_obligations, has_deadlines, has_monetary_values
- "language": ISO 639-1 language code (e.g. "en")
- "summary": one-sentence summary of the chunk, max 30 words

Return ONLY valid JSON. No markdown, no explanation.

{warning}

{document}"""


class LLMEnricher(Enricher):
    """LLM-powered enricher for deep semantic extraction from document chunks.

    Extracts named entities, keyphrases, relationships, semantic flags,
    language, and a one-sentence summary via a single LLM call per chunk.

    The ``llm_fn`` contract matches blue-data's ``MetaData`` enricher — any
    callable with the signature ``(prompt: str) -> str`` works:

        from anthropic import Anthropic
        client = Anthropic()
        llm_fn = lambda p: client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": p}],
        ).content[0].text

        enricher = LLMEnricher(llm_fn=llm_fn, fallback=BasicEnricher())

    Args:
        llm_fn: Callable that takes a prompt string and returns the LLM response.
        fallback: Enricher to call when the LLM fails or returns unparseable
            output. When ``None`` a minimal structure (empty entities, computed
            word/sentence counts, empty optional fields) is returned instead.
        max_chunk_chars: Chunk text is truncated to this length before being
            sent to the LLM. Prevents unbounded token spend on large chunks.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        fallback: Optional[Enricher] = None,
        max_chunk_chars: int = 2000,
    ):
        self._llm_fn = llm_fn
        self._fallback = fallback
        self._max_chunk_chars = max_chunk_chars

    def enrich(self, text: str) -> Dict[str, Any]:
        truncated = text[: self._max_chunk_chars]
        try:
            raw = self._llm_fn(_PROMPT.format(
                warning=UNTRUSTED_DOCUMENT_WARNING,
                document=wrap_document(truncated),
            ))
            parsed = _parse_json(raw)
        except Exception:
            return self._fallback.enrich(text) if self._fallback else _minimal(text)

        entities: List[Dict[str, Any]] = [
            e for e in parsed.get("entities", [])
            if isinstance(e, dict) and "text" in e and "type" in e
        ]
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        return {
            "entities": entities,
            "semantic_hints": {
                "word_count": len(text.split()),
                "sentence_count": len(sentences),
                "keyphrases": parsed.get("keyphrases", []),
                "relationships": parsed.get("relationships", []),
                "semantic_flags": parsed.get("semantic_flags", {}),
                "language": parsed.get("language", ""),
                "summary": parsed.get("summary", ""),
            },
        }


def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM response")
    return json.loads(match.group())


def _minimal(text: str) -> Dict[str, Any]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return {
        "entities": [],
        "semantic_hints": {
            "word_count": len(text.split()),
            "sentence_count": len(sentences),
            "keyphrases": [],
            "relationships": [],
            "semantic_flags": {},
            "language": "",
            "summary": "",
        },
    }
