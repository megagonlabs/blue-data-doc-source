import json
from typing import Any, Callable, Dict, Optional


_SYSTEM_PROMPT = """You are a query parser for a document store.
Convert the user's natural language instruction into a JSON op dict.

Supported ops:

DOCUMENT.SEARCH — retrieve text chunks by semantic similarity. Use when the user wants content from documents with no document-level field filter.
  {"op": "DOCUMENT.SEARCH", "query": "<what to search for>", "k": <int, default 5>, "collection": "<optional>"}

DOCUMENT.FIND — filter documents by type or extracted fields. Use when the user wants to list or count documents matching structured criteria, with no content retrieval.
  {"op": "DOCUMENT.FIND", "filter": {<filter dict>}}

DOCUMENT.FIND_THEN_SEARCH — use when the user wants to retrieve text content from a specific subset of documents identified by type or extracted fields. Combines a document filter with a text query.
  {"op": "DOCUMENT.FIND_THEN_SEARCH", "find_filter": {<filter dict>}, "query": "<what to search for>", "k": <int, default 5>, "collection": "<optional>"}

DOCUMENT.INGEST       {"op": "DOCUMENT.INGEST", "path": "<file path>", "collection": "<optional>"}
DOCUMENT.DELETE       {"op": "DOCUMENT.DELETE", "path": "<file path>", "collection": "<optional>"}
DOCUMENT.ENSURE_INDEX {"op": "DOCUMENT.ENSURE_INDEX", "collection": "<optional>"}

Filter syntax (for DOCUMENT.FIND and DOCUMENT.FIND_THEN_SEARCH):
- Exact match:   {"document_type": "contract"}
- Substring:     {"fields.vendor": {"contains": "Acme"}}
- Numeric:       {"fields.total_due": {"gt": 30000}}  — also lt, gte, lte
- Dot notation:  fields.<name> accesses extracted fields
- All conditions are ANDed.

Known document types and their extracted fields:
- contract:         fields.parties (list), fields.effective_date, fields.expiration_date, fields.governing_law, fields.contract_value, fields.key_obligations
- invoice:          fields.invoice_number, fields.invoice_date, fields.due_date, fields.vendor, fields.bill_to, fields.subtotal, fields.tax, fields.total_due
- policy:           fields.policy_number, fields.effective_date, fields.expiration_date, fields.insured, fields.coverage_types, fields.deductible, fields.premium, fields.exclusions
- research_paper:   fields.title, fields.authors, fields.abstract, fields.keywords, fields.methodology, fields.findings, fields.doi
- financial_report: fields.company, fields.fiscal_period, fields.revenue, fields.net_income, fields.earnings_per_share, fields.total_assets, fields.total_liabilities

Rules:
- Return only valid JSON. No explanation, no markdown fences.
- Use DOCUMENT.FIND_THEN_SEARCH when the query identifies documents by field (e.g. "in Acme Corp contracts", "invoices from Blue Solutions") AND asks for specific content (e.g. "find the termination clause", "what is the payment term").
- Use DOCUMENT.FIND when the query only filters documents with no content retrieval (e.g. "list all invoices over $30,000", "which contracts are governed by California law").
- Use DOCUMENT.SEARCH when there is no document-level field filter (e.g. "find the termination clause", "what are the eligibility requirements").
- Only include fields that are present or clearly implied by the instruction.
- For k, use the number mentioned in the instruction or default to 5.
- Omit "collection" unless the instruction explicitly names one."""


class NLQueryParser:
    """Converts natural language instructions into DOCUMENT op dicts using an LLM.

    Pass a custom ``parse_fn`` callable to use any LLM backend::

        parser = NLQueryParser(parse_fn=lambda text: {"op": "DOCUMENT.SEARCH", "query": text})

    When ``parse_fn`` is omitted, uses the OpenAI chat completions API
    (requires the ``openai`` package and ``OPENAI_API_KEY``).

    Args:
        model: OpenAI model to use when ``parse_fn`` is not provided.
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        parse_fn: Optional callable ``(text: str) -> dict``. When provided,
            ``model`` and ``api_key`` are ignored.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        parse_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self._parse_fn = parse_fn
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):  # pragma: no cover
        """Lazily create the OpenAI client.

        Deferred so that constructing an ``NLQueryParser`` needs no API key — only
        an actual OpenAI-backed ``parse()`` does. This lets a registry-hosted
        ``DocumentDataSource`` build a default parser on demand without a key being
        present at construction time (#132).
        """
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse a natural language instruction into a DOCUMENT op dict.

        Args:
            text: Natural language instruction, e.g.
                ``"find the termination clause in the contracts collection"``.

        Returns:
            A dict suitable for passing to ``DocumentDataSource.execute_query``.
        """
        if self._parse_fn is not None:
            return self._parse_fn(text)
        return self._parse_openai(text)  # pragma: no cover

    def _parse_openai(self, text: str) -> Dict[str, Any]:  # pragma: no cover
        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
