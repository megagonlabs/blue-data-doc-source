from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class CoverageReport:
    """Coverage statistics for a :func:`find_records` call.

    Attributes:
        documents_considered: Total documents in the store evaluated against the filter.
        documents_matched: Documents that passed every filter condition.
        documents_missing_field: Documents where at least one filter-specified
            ``fields.*`` key was absent — these could not be evaluated and may
            be false negatives.
        field_coverage_rate: ``(documents_considered - documents_missing_field)
            / documents_considered``. ``1.0`` means every document had the
            queried field(s); lower values indicate extraction gaps.
    """

    documents_considered: int
    documents_matched: int
    documents_missing_field: int
    field_coverage_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_considered": self.documents_considered,
            "documents_matched": self.documents_matched,
            "documents_missing_field": self.documents_missing_field,
            "field_coverage_rate": round(self.field_coverage_rate, 4),
        }


@dataclass
class FindResult:
    """Return type of :func:`find_records` when ``coverage=True``.

    Iterable over :attr:`records` so existing code that does
    ``for r in pipeline.find_records(...)`` continues to work without
    modification when coverage reporting is added.

    Attributes:
        records: Matched DocumentRecords — same list as the plain return value.
        status: ``"ok"`` when all documents had the queried field(s);
            ``"partial"`` when some documents are missing a field;
            ``"no_data"`` when the store is empty.
        coverage: Detailed counts — see :class:`CoverageReport`.
        warning: Human-readable explanation when ``status == "partial"``.
    """

    records: List[Dict[str, Any]]
    status: str
    coverage: CoverageReport
    warning: Optional[str] = None

    # ------------------------------------------------------------------
    # Iterable interface — backward-compatible with plain list callers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "records": self.records,
            "status": self.status,
            "coverage": self.coverage.to_dict(),
        }
        if self.warning:
            d["warning"] = self.warning
        return d
