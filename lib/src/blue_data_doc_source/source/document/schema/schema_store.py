from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from blue_data.source.json.json_source import JSONSource


class RecordStore(ABC):
    """Pluggable storage backend for per-document schema records.

    Each record is the dict produced by the classification + extraction
    pipeline: ``{document_id, document_type, confidence, classifier, fields}``.
    """

    @abstractmethod
    def upsert(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def find(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return all records matching *filter*.

        Filter keys support dot notation for nested access::

            {"document_type": "contract"}
            {"fields.governing_law": "California"}
            {"fields.total_due": {"gt": 30000}}
            {"document_type": "invoice", "fields.vendor": {"contains": "Acme"}}

        Supported operators (pass as a dict value):
            ``contains`` — case-insensitive substring match
            ``lt``, ``gt``, ``lte``, ``gte`` — numeric (strips $,€,£,¥ and
            commas before parsing) or lexicographic if not numeric

        All conditions are ANDed. Pass ``None`` or ``{}`` to return every record.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str) -> None:
        raise NotImplementedError


class MemoryRecordStore(RecordStore):
    """In-process schema store — default, zero dependencies."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, record: Dict[str, Any]) -> None:
        self._store[record["document_id"]] = record

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(document_id)

    def find(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [r for r in self._store.values() if _matches(r, filter)]

    def delete(self, document_id: str) -> None:
        self._store.pop(document_id, None)


class FileRecordStore(RecordStore):
    """JSON-file-backed schema store — survives process restarts.

    Writes atomically (write to ``<path>.tmp`` then ``os.replace``) so a
    crash mid-write never corrupts the file.

    Args:
        path: Path to the JSON file. Created on first write if absent.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            if content:
                self._store = json.loads(content)

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)

    def upsert(self, record: Dict[str, Any]) -> None:
        self._store[record["document_id"]] = record
        self._save()

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(document_id)

    def find(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [r for r in self._store.values() if _matches(r, filter)]

    def delete(self, document_id: str) -> None:
        if document_id in self._store:
            del self._store[document_id]
            self._save()


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _matches(record: Dict[str, Any], filter: Optional[Dict[str, Any]]) -> bool:
    if not filter:
        return True
    return all(_check(record, key, condition) for key, condition in filter.items())


def _get_nested(record: Dict[str, Any], key: str) -> Any:
    """Resolve a dot-separated key: 'fields.governing_law' → record['fields']['governing_law']."""
    head, _, tail = key.partition(".")
    val = record.get(head)
    if not tail or not isinstance(val, dict):
        return val
    return val.get(tail)


def _try_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[$€£¥,]", "", value).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _check(record: Dict[str, Any], key: str, condition: Any) -> bool:
    value = _get_nested(record, key)
    if value is None:
        return False

    if not isinstance(condition, dict):
        # Exact match — case-insensitive for strings
        if isinstance(value, str) and isinstance(condition, str):
            return value.lower() == condition.lower()
        return value == condition

    # Operator dict
    for op, operand in condition.items():
        if op == "contains":
            if not (isinstance(value, str) and str(operand).lower() in value.lower()):
                return False
        elif op in ("lt", "gt", "lte", "gte"):
            nv, no = _try_numeric(value), _try_numeric(operand)
            if nv is not None and no is not None:
                if op == "lt"  and not nv <  no: return False
                if op == "gt"  and not nv >  no: return False
                if op == "lte" and not nv <= no: return False
                if op == "gte" and not nv >= no: return False
            else:
                sv, so = str(value), str(operand)
                if op == "lt"  and not sv <  so: return False
                if op == "gt"  and not sv >  so: return False
                if op == "lte" and not sv <= so: return False
                if op == "gte" and not sv >= so: return False
    return True


# ---------------------------------------------------------------------------
# JSONSource-backed adapter
# ---------------------------------------------------------------------------

class JsonSourceRecordStore(RecordStore):
    """RecordStore backed by any blue-data JSONSource.

    Accepts ``MemoryJSONSource``, ``FileJSONSource``, ``RedisJSONSource``, or
    any other ``JSONSource`` implementation. Each schema record is stored at its
    ``document_id`` as the key. The ``find()`` filter logic runs in Python after
    fetching all records from the source.

    Example (Redis-backed, multi-process-safe)::

        from blue_data.source.json.redis.redis_json_source import RedisJSONSource
        from blue_data_doc_source import JsonSourceRecordStore

        store = JsonSourceRecordStore(
            RedisJSONSource(
                name="doc_schemas",
                properties={"namespace": "doc_schemas"},
            )
        )
    """

    def __init__(self, json_source: "JSONSource") -> None:
        self._source = json_source

    def upsert(self, record: Dict[str, Any]) -> None:
        self._source.set(record["document_id"], record)

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._source.get(document_id)

    def find(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        all_data = self._source.get(None) or {}
        records = all_data.values() if isinstance(all_data, dict) else []
        return [r for r in records if isinstance(r, dict) and _matches(r, filter)]

    def delete(self, document_id: str) -> None:
        self._source.delete(document_id)
