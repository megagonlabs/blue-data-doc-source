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

# Reserved key used to surface the validation outcome on the returned dict in non-strict
# mode. A caller reads ``result.get("_validation")`` to detect a degraded result.
_VALIDATION_KEY = "_validation"

# JSON Schema primitive type name -> acceptable Python type(s).
_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class ExtractionValidationError(ValueError):
    """Raised when StructuredFieldExtractor runs in strict mode (the default) and the
    LLM output is not valid JSON, is not a JSON object, or does not satisfy the schema.

    Carries the list of validation error messages on ``.errors``. Strict mode replaces
    the old behaviour of silently returning the raw, unvalidated dict (#64).
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


def _is_pydantic(schema: Any) -> bool:
    return isinstance(schema, type) and hasattr(schema, "model_json_schema")


def _schema_label(schema: Any) -> str:
    if isinstance(schema, dict):
        return "JSON Schema"
    return getattr(schema, "__name__", type(schema).__name__)


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


def _try_parse_json(raw: str) -> tuple[Any, bool]:
    """Parse *raw* as JSON, tolerating a markdown code fence. Returns ``(value, ok)``."""
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError:
            return None, False


def _parse_json(raw: str) -> dict:
    """Back-compat helper: parse JSON, returning ``{"raw_response": raw}`` on failure."""
    value, ok = _try_parse_json(raw)
    return value if ok and isinstance(value, dict) else {"raw_response": raw}


def _validate_json_schema(data: dict, schema: dict) -> list[str]:
    """Validate *data* against a JSON Schema dict, returning a list of error messages
    (empty when valid). Uses ``jsonschema`` when installed for full validation; falls
    back to a minimal required-key + top-level type check otherwise."""
    try:
        import jsonschema  # type: ignore[import]
    except ImportError:
        jsonschema = None

    if jsonschema is not None:
        validator = jsonschema.Draft7Validator(schema)
        return [e.message for e in sorted(validator.iter_errors(data), key=lambda e: list(e.path))]

    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in data:
            errors.append(f"'{key}' is a required property")
    for name, info in schema.get("properties", {}).items():
        if name not in data:
            continue
        expected = info.get("type")
        py = _JSON_TYPE_MAP.get(expected) if isinstance(expected, str) else None
        if py is None:
            continue
        val = data[name]
        # bool is a subclass of int — reject it where a number/integer is expected.
        if expected in ("integer", "number") and isinstance(val, bool):
            errors.append(f"{name}: {val!r} is not of type {expected!r}")
        elif not isinstance(val, py):
            errors.append(f"{name}: {val!r} is not of type {expected!r}")
    return errors


def _typeddict_required_keys(schema: Any, hints: dict) -> set[str]:
    required = getattr(schema, "__required_keys__", None)
    if required is not None:
        return set(required)
    return set(hints.keys())  # total=True TypedDict / annotated class — all required


def _validate_typeddict(data: dict, schema: Any) -> list[str]:
    """Validate *data* against a TypedDict / annotated class: required keys present and,
    for plain builtin-typed fields, values of the right type. Generic/Optional hints are
    checked for presence only."""
    try:
        hints = get_type_hints(schema)
    except Exception:
        hints = dict(getattr(schema, "__annotations__", {}))

    errors: list[str] = []
    for key in _typeddict_required_keys(schema, hints):
        if key not in data:
            errors.append(f"'{key}' is a required field")

    for name, hint in hints.items():
        if name not in data or not isinstance(hint, type):
            continue  # only type-check plain builtin hints (str/int/float/bool/…)
        val = data[name]
        if hint is float and isinstance(val, int) and not isinstance(val, bool):
            continue  # an int is acceptable where a float is expected
        if hint in (int, float) and isinstance(val, bool):
            errors.append(f"{name}: expected {hint.__name__}, got bool")
        elif not isinstance(val, hint):
            errors.append(f"{name}: expected {hint.__name__}, got {type(val).__name__}")
    return errors


class StructuredFieldExtractor(FieldExtractor):
    """Field extractor driven by a caller-supplied schema.

    Accepts a Pydantic model class, TypedDict, or a JSON Schema dict. Generates the
    extraction prompt automatically from the schema definition — no manual field spec
    string required.

    Validation (#64). Because this is *typed* extraction, the LLM output is validated
    against the schema for every schema kind — Pydantic via ``model_validate``, JSON
    Schema via ``jsonschema`` (or a built-in fallback), TypedDict via its type hints. In
    ``strict`` mode (the default) a parse failure, a non-object response, or any schema
    violation raises :class:`ExtractionValidationError` rather than silently returning the
    raw dict. With ``strict=False`` the best-effort dict is returned with a reserved
    ``"_validation"`` key — ``{"status": "invalid", "errors": [...]}`` — so a caller can
    detect the degradation instead of being handed unvalidated values.

    For Pydantic models a valid result is dumped in JSON mode so date/datetime/Decimal
    fields become JSON-native primitives a JSON record store can serialise (#89).

    Args:
        schema: A Pydantic model class, TypedDict class, or JSON Schema dict.
        model: OpenAI model to use. Default ``"gpt-4o-mini"``.
        sample_chars: Characters of document text sent to the LLM. Default 4000.
        strict: Raise :class:`ExtractionValidationError` on invalid output instead of
            returning it annotated. Default ``True``.

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
        strict: bool = True,
    ) -> None:
        self._schema = schema
        self._model = model
        self._sample_chars = sample_chars
        self._strict = strict
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

    @staticmethod
    def _pydantic_errors(exc: Exception) -> list[str]:
        errs = getattr(exc, "errors", None)
        if callable(errs):
            out = []
            for e in exc.errors():  # type: ignore[attr-defined]
                loc = ".".join(str(p) for p in e.get("loc", ()))
                msg = e.get("msg", "invalid")
                out.append(f"{loc}: {msg}" if loc else msg)
            if out:
                return out
        return [str(exc)]

    def _finalize(self, data: dict, errors: list[str]) -> dict:
        """Apply the strict/non-strict policy to a validation outcome."""
        if not errors:
            return data
        if self._strict:
            raise ExtractionValidationError(
                f"{_schema_label(self._schema)} extraction failed validation: "
                + "; ".join(errors),
                errors=errors,
            )
        warnings.warn(
            "StructuredFieldExtractor output failed schema validation "
            f"({_schema_label(self._schema)}): " + "; ".join(errors),
            stacklevel=2,
        )
        out = dict(data)
        out[_VALIDATION_KEY] = {"status": "invalid", "errors": errors}
        return out

    def _validate(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return self._finalize(
                {"raw_response": data},
                [f"expected a JSON object, got {type(data).__name__}"],
            )
        if self._pydantic:
            try:
                # mode="json" coerces date/datetime/Decimal/etc. to JSON-native
                # primitives so a JSON record store can serialise the fields (#89).
                coerced = self._schema.model_validate(data).model_dump(
                    mode="json", exclude_none=True
                )
            except Exception as exc:
                return self._finalize(data, self._pydantic_errors(exc))
            return self._finalize(coerced, [])
        if isinstance(self._schema, dict):
            return self._finalize(data, _validate_json_schema(data, self._schema))
        return self._finalize(data, _validate_typeddict(data, self._schema))

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
        parsed, ok = _try_parse_json(raw)
        if not ok:
            return self._finalize({"raw_response": raw}, ["response was not valid JSON"])
        return self._validate(parsed)
