from __future__ import annotations

import json
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


class LLMFieldExtractor(FieldExtractor):
    """OpenAI-backed field extractor — optional dependency.

    Requires ``openai>=1.0`` to be installed and ``OPENAI_API_KEY`` set.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        sample_chars: int = 4000,
    ) -> None:
        self._model = model
        self._sample_chars = sample_chars
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
        field_spec = _FIELD_SPECS.get(document_type, "Extract any structured metadata you can find.")
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
