from __future__ import annotations

from .base import DocumentClassifier

# Signal phrases scored against the first ~500 words of the document.
# Longer phrases score the same as single keywords — choose signals that are
# discriminative for the type rather than exhaustive.
_TYPE_SIGNALS: dict[str, list[str]] = {
    "contract": [
        "agreement", "parties", "governing law", "termination", "indemnification",
        "whereas", "hereby", "obligations", "warranties", "confidentiality",
        "intellectual property", "liability", "breach", "remedy", "arbitration",
        "force majeure", "entire agreement", "severability",
    ],
    "invoice": [
        "invoice", "due date", "amount due", "bill to", "line item",
        "payment", "subtotal", "tax", "total", "remit to",
        "purchase order", "vendor", "quantity", "unit price", "invoice number",
    ],
    "policy": [
        "policy", "eligibility", "coverage", "effective date",
        "benefits", "deductible", "premium", "exclusion", "claim",
        "insured", "beneficiary", "renewal", "cancellation", "copay",
    ],
    "research_paper": [
        "abstract", "methodology", "references", "conclusion", "hypothesis",
        "introduction", "literature review", "findings", "discussion",
        "figure", "citation", "doi", "keywords", "we propose", "our approach",
    ],
    "financial_report": [
        "revenue", "net income", "balance sheet", "earnings per share",
        "fiscal year", "operating income", "cash flow", "liabilities",
        "assets", "equity", "dividend", "quarterly", "annual report",
        "ebitda", "gross margin",
    ],
}

_MIN_CONFIDENCE: float = 0.15
_SAMPLE_WORDS: int = 500


class KeywordClassifier(DocumentClassifier):
    """Rule-based document classifier — no API cost.

    Scores each document type by counting how many of its signal phrases appear
    in the first ``sample_words`` words of the document. The type with the
    highest match ratio wins; if the winner's ratio is below ``min_confidence``,
    ``"unknown"`` is returned.
    """

    def __init__(
        self,
        min_confidence: float = _MIN_CONFIDENCE,
        sample_words: int = _SAMPLE_WORDS,
    ) -> None:
        self._min_confidence = min_confidence
        self._sample_words = sample_words

    def classify(self, text: str) -> tuple[str, float]:
        words = text.split()
        sample = " ".join(words[: self._sample_words]).lower()

        best_type = "unknown"
        best_score = 0.0

        for doc_type, signals in _TYPE_SIGNALS.items():
            matches = sum(1 for s in signals if s in sample)
            score = matches / len(signals)
            if score > best_score:
                best_score = score
                best_type = doc_type

        if best_score < self._min_confidence:
            return "unknown", round(best_score, 4)

        return best_type, round(best_score, 4)
