from __future__ import annotations

import json
import warnings
from typing import Any

from .base import FieldExtractor

_FIELD_SPECS: dict[str, str] = {
    "contract": (
        "Extract: parties (list of strings), effective_date (ISO date string), "
        "expiration_date (ISO date string), governing_law (jurisdiction string), "
        "contract_value (string with currency), key_obligations (list of strings)."
    ),
    "invoice": (
        "Extract: invoice_number (string), invoice_date (ISO date string), "
        "due_date (ISO date string), vendor (string), bill_to (string), "
        "line_items (list of {description, quantity, unit_price, total}), "
        "subtotal (string), tax (string), total_due (string)."
    ),
    "policy": (
        "Extract: policy_number (string), effective_date (ISO date string), "
        "expiration_date (ISO date string), insured (string), "
        "coverage_types (list of strings), deductible (string), "
        "premium (string), exclusions (list of strings)."
    ),
    "research_paper": (
        "Extract: title (string), authors (list of strings), abstract (string), "
        "keywords (list of strings), methodology (string), "
        "findings (string), doi (string)."
    ),
    "financial_report": (
        "Extract: company (string), fiscal_period (string), "
        "revenue (string), net_income (string), "
        "earnings_per_share (string), total_assets (string), "
        "total_liabilities (string), highlights (list of strings)."
    ),
}

_DEFAULT_PROMPT = (
    "Extract structured metadata from the following document text. "
    "Return only a valid JSON object — no prose, no markdown fences.\n\n"
    "{field_spec}\n\n"
    "Omit fields you cannot determine. Do not invent values.\n\n"
    "Document text:\n{text}"
)

_GENERIC_SPEC = "Extract any structured metadata you can find."


class LLMFieldExtractor(FieldExtractor):
    """OpenAI-backed field extractor — optional dependency.

    Requires ``openai>=1.0`` to be installed and ``OPENAI_API_KEY`` set.

    Args:
        model: OpenAI model to use for extraction. Default ``"gpt-4o-mini"``.
        sample_chars: Maximum characters from the start of the document sent to
            the LLM. Default 4000.
        extra_types: Additional document types and their field extraction specs,
            supplied as a mapping of ``{type_name: field_spec_string}``. The
            field spec string is a plain-English instruction telling the LLM
            what to extract, in the same format as the built-in specs. Merged
            with the built-in types; user-supplied types take precedence on
            name collision.
        hint_fields: Preferred field names to use when extracting from an
            unrecognised document type (i.e. when no type-specific spec
            matches). The LLM is instructed to use these exact names for any
            data it finds that corresponds to them, reducing naming variance
            across documents. Additional fields not in this list may still be
            extracted. Has no effect when ``document_type`` matches a built-in
            or ``extra_types`` spec, since those specs already define field
            names precisely.

    Example::

        extractor = LLMFieldExtractor(
            extra_types={
                "lease": (
                    "Extract: lessor (string), lessee (string), "
                    "monthly_rent (string with currency), lease_term (string), "
                    "security_deposit (string), premises_address (string)."
                ),
                "employment": (
                    "Extract: employer (string), employee (string), "
                    "job_title (string), start_date (ISO date string), "
                    "salary (string with currency), jurisdiction (string)."
                ),
            }
        )

        # Stabilise field names during schema discovery (no known type yet):
        extractor = LLMFieldExtractor(
            hint_fields=["vendor", "total_due", "date", "status"],
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        sample_chars: int = 4000,
        extra_types: dict[str, str] | None = None,
        hint_fields: list[str] | None = None,
    ) -> None:
        self._model = model
        self._sample_chars = sample_chars
        self._field_specs = {**_FIELD_SPECS, **(extra_types or {})}
        self._hint_fields = hint_fields
        if hint_fields:
            names = ", ".join(hint_fields)
            self._generic_spec = (
                f"{_GENERIC_SPEC} "
                f"When you find values that correspond to any of the following concepts, "
                f"use exactly these field names: {names}. "
                f"You may extract additional fields beyond this list."
            )
        else:
            self._generic_spec = _GENERIC_SPEC
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for LLMFieldExtractor. "
                    "Install with: pip install 'blue-data-doc-source[openai]'"
                ) from exc
            self._client = OpenAI()
        return self._client

    def extract(self, text: str, document_type: str) -> dict:
        if len(text) > self._sample_chars:
            warnings.warn(
                f"LLMFieldExtractor truncated the document to the first "
                f"{self._sample_chars} of {len(text)} characters; fields that appear "
                f"only after that point will be missed (see issue #63). Increase "
                f"sample_chars to cover more of the document.",
                stacklevel=2,
            )
        field_spec = self._field_specs.get(document_type, self._generic_spec)
        prompt = _DEFAULT_PROMPT.format(
            field_spec=field_spec,
            text=text[: self._sample_chars],
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Best-effort: strip common markdown fences and retry
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"raw_response": raw}
