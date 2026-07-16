import json
from typing import Any, Callable, Dict, Optional


_SYSTEM_PROMPT = """You are a query parser for a document store.
Convert the user's natural language instruction into a JSON op dict.

Supported ops:
- DOCUMENT.SEARCH       {"op": "DOCUMENT.SEARCH", "query": "<what to search for>", "k": <int, default 5>, "collection": "<optional>"}
- DOCUMENT.INGEST       {"op": "DOCUMENT.INGEST", "path": "<file path>", "collection": "<optional>"}
- DOCUMENT.DELETE       {"op": "DOCUMENT.DELETE", "path": "<file path>", "collection": "<optional>"}
- DOCUMENT.ENSURE_INDEX {"op": "DOCUMENT.ENSURE_INDEX", "collection": "<optional>"}

Rules:
- Return only valid JSON. No explanation, no markdown fences.
- Infer the op from the instruction. When in doubt, use DOCUMENT.SEARCH.
- Only include fields that are present or clearly implied by the instruction.
- For k, use the number mentioned in the instruction or default to 5."""


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
        if parse_fn is None:  # pragma: no cover
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._model = model

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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
