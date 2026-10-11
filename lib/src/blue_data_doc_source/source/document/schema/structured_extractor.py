from __future__ import annotations

import json
import warnings
from typing import Any, get_type_hints

from .base import FieldExtractor

_EXTRACTION_PROMPT = (
    "Extract structured metadata from the following document text. "
    "Return only a valid JSON object — no prose, no markdown fences.\n\n"
    "{field_spec}\n\n"
    "Omit fields you cannot determine. Do not invent values.\n\n"
    "Document text:\n{text}"
)


def _is_pydantic(schema: Any) -> bool:
    return isinstance(schema, type) and hasattr(schema, "model_json_schema")


def _field_spec_from_schema(schema: Any) -> str:
    """Build a plain-English field list from a Pydantic model, TypedDict, or JSON Schema dict."""
    if isinstance(schema, dict):
        props = schema.get("properties", {})
        parts = [f"{name} ({info.get('type', 'any')})" for name, info in props.items()]

    elif _is_pydantic(schema):
        js = schema.model_json_schema()
        props = js.get("properties", {})
        parts = []
        for name, info in props.items():
            # Pydantic v2 uses anyOf for Optional fields
            type_str = info.get("type") or (info.get("anyOf", [{}])[0].get("type", "any"))
            parts.append(f"{name} ({type_str})")

    else:
        # TypedDict or annotated class
        try:
            hints = get_type_hints(schema)
        except Exception:
            hints = getattr(schema, "__annotations__", {})
        parts = [f"{name} ({getattr(t, '__name__', str(t))})" for name, t in hints.items()]

    return ("Extract these fields: " + ", ".join(parts) + ".") if parts else "Extract structured fields."


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_response": raw}


class StructuredFieldExtractor(FieldExtractor):
    """Field extractor driven by a caller-supplied schema.

    Accepts a Pydantic model class, TypedDict, or a JSON Schema dict.
    Generates the extraction prompt automatically from the schema definition —
    no manual field spec string required.

    For Pydantic models, validates and coerces the LLM output via
    ``model_validate`` before returning; falls back to the raw parsed dict if
    validation fails.

    Args:
        schema: A Pydantic model class, TypedDict class, or JSON Schema dict.
        model: OpenAI model to use. Default ``"gpt-4o-mini"``.
        sample_chars: Characters of document text sent to the LLM. Default 4000.

    Example::

        from pydantic import BaseModel

        class EmploymentContract(BaseModel):
            employee_name: str
            start_date: str
            base_salary: str
            notice_period: str
            non_compete: bool

        pipeline = DocumentPipeline(
            field_extractor=StructuredFieldExtractor(schema=EmploymentContract),
        )
    """

    def __init__(
        self,
        schema: type | dict,
        model: str = "gpt-4o-mini",
        sample_chars: int = 4000,
    ) -> None:
        self._schema = schema
        self._model = model
        self._sample_chars = sample_chars
        self._field_spec = _field_spec_from_schema(schema)
        self._pydantic = _is_pydantic(schema)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for StructuredFieldExtractor. "
                    "Install with: pip install 'blue-data-doc-source[openai]'"
                ) from exc
            self._client = OpenAI()
        return self._client

    def _validate(self, data: dict) -> dict:
        if not self._pydantic:
            return data
        try:
            return self._schema.model_validate(data).model_dump(exclude_none=True)
        except Exception:
            return data

    def extract(self, text: str, document_type: str) -> dict:
        if len(text) > self._sample_chars:
            warnings.warn(
                f"StructuredFieldExtractor truncated the document to the first "
                f"{self._sample_chars} of {len(text)} characters; fields that appear "
                f"only after that point will be missed (see issue #63). Increase "
                f"sample_chars to cover more of the document.",
                stacklevel=2,
            )
        prompt = _EXTRACTION_PROMPT.format(
            field_spec=self._field_spec,
            text=text[: self._sample_chars],
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        return self._validate(_parse_json(raw))
