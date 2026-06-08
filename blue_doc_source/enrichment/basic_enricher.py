import re
from collections import Counter
from typing import Any, Dict, List

from .base import Enricher

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "as", "if", "then", "that", "this",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "my", "me", "us", "who", "which",
    "what", "when", "where", "how", "all", "each", "every", "any", "more",
    "also", "into", "than", "about", "up", "out", "over", "after",
}

_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"          # 01/15/2024 or 1-5-24
    r"|\d{4}[-/]\d{2}[-/]\d{2}"                      # 2024-01-15
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?"
    r"|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?"
    r"|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_MONEY_PATTERN = re.compile(r"[$€£¥]\s*\d+(?:[,\d]*)?(?:\.\d+)?|\b\d+(?:[,\d]*)?\s*(?:dollars?|euros?|pounds?)\b", re.IGNORECASE)
_PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


class BasicEnricher(Enricher):
    def __init__(self, top_k_keywords: int = 10):
        self.top_k_keywords = top_k_keywords

    def enrich(self, text: str) -> Dict[str, Any]:
        entities = self._extract_entities(text)
        semantic_hints = self._extract_semantic_hints(text)
        return {"entities": entities, "semantic_hints": semantic_hints}

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []

        for match in _DATE_PATTERN.finditer(text):
            entities.append({"text": match.group().strip(), "type": "DATE"})

        for match in _EMAIL_PATTERN.finditer(text):
            entities.append({"text": match.group().strip(), "type": "EMAIL"})

        for match in _PERCENT_PATTERN.finditer(text):
            entities.append({"text": match.group().strip(), "type": "PERCENT"})

        for match in _MONEY_PATTERN.finditer(text):
            entities.append({"text": match.group().strip(), "type": "MONEY"})

        seen = set()
        for match in _PROPER_NOUN_PATTERN.finditer(text):
            phrase = match.group().strip()
            if phrase not in seen:
                seen.add(phrase)
                entities.append({"text": phrase, "type": "PROPER_NOUN"})

        return entities

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [t for t in tokens if t not in _STOPWORDS]
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(self.top_k_keywords)]

    def _extract_semantic_hints(self, text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "has_questions": "?" in text,
            "has_numbers": bool(re.search(r"\b\d+\b", text)),
            "keywords": self._extract_keywords(text),
        }
