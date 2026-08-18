from abc import ABC, abstractmethod


class DocumentClassifier(ABC):
    """Classify a document into a known type.

    Returns a ``(document_type, confidence)`` tuple where ``confidence`` is a
    float in [0, 1]. Implementations should return ``("unknown", 0.0)`` when no
    type can be determined with sufficient confidence.
    """

    @abstractmethod
    def classify(self, text: str) -> tuple[str, float]:
        raise NotImplementedError


class FieldExtractor(ABC):
    """Extract typed fields from a document given its classified type.

    Returns a plain dict whose keys are type-specific field names. Absent or
    unextractable fields should be omitted rather than set to None.
    """

    @abstractmethod
    def extract(self, text: str, document_type: str) -> dict:
        raise NotImplementedError
