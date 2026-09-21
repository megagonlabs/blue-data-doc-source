"""Extend installed blue_data test doubles with missing capabilities."""
from __future__ import annotations

import math
import re

from blue_data.source.graph.memory_graph_source import MemoryGraphSource
from blue_data.source.vector.memory_vector_source import MemoryVectorSource


# ---------------------------------------------------------------------------
# MemoryVectorSource — extended filter support
# ---------------------------------------------------------------------------

def _apply_filter_extended(self, docs: dict, filter_expr: str) -> dict:
    """Support tag, numeric-range, and compound AND filters in addition to '*'."""
    if not filter_expr or filter_expr.strip() == "*":
        return docs

    expr = filter_expr.strip()

    # Compound AND: (expr1) (expr2) ...
    if expr.startswith("("):
        parts = re.findall(r"\(([^)]+)\)", expr)
        result = dict(docs)
        for part in parts:
            result = _apply_filter_extended(self, result, part.strip())
        return result

    # Legacy @id:{value} — match on dict key (original behaviour)
    m = re.match(r"@id:\{(.+)\}$", expr)
    if m:
        doc_id = m.group(1)
        return {k: v for k, v in docs.items() if k == doc_id}

    # Tag filter: @field:{value}  (may have \. \/ \ -escaped characters)
    m = re.match(r"@(\w+):\{(.+)\}$", expr)
    if m:
        field, raw_val = m.group(1), m.group(2)
        value = re.sub(r"\\(.)", r"\1", raw_val)
        return {k: v for k, v in docs.items() if str(v.get(field, "")) == value}

    # Numeric range: @field:[min max]
    m = re.match(r"@(\w+):\[(.+?) (.+?)\]$", expr)
    if m:
        field = m.group(1)
        lo = -math.inf if m.group(2) == "-inf" else float(m.group(2))
        hi = math.inf if m.group(3) == "+inf" else float(m.group(3))
        result = {}
        for k, v in docs.items():
            val = v.get(field)
            if val is not None:
                try:
                    if lo <= float(val) <= hi:
                        result[k] = v
                except (TypeError, ValueError):
                    pass
        return result

    self.logger.warning(f"MemoryVectorSource: unsupported filter '{expr}' — returning all docs")
    return docs


MemoryVectorSource._apply_filter = _apply_filter_extended


# ---------------------------------------------------------------------------
# MemoryGraphSource — match_edges / match_nodes helpers used by unit tests
# ---------------------------------------------------------------------------

def _match_edges(
    self,
    database: str,
    edge_type: str | None = None,
    source_id: str | None = None,
    target_id: str | None = None,
) -> list[dict]:
    return [
        e for e in self._edges[database].values()
        if (edge_type is None or e["edge_type"] == edge_type)
        and (source_id is None or e["source_id"] == source_id)
        and (target_id is None or e["target_id"] == target_id)
    ]


def _match_nodes(
    self,
    database: str,
    label: str,
    props: dict | None = None,
) -> list[dict]:
    results = []
    for node in self._nodes[database].values():
        if node["label"] != label:
            continue
        if props and not all(node["properties"].get(k) == v for k, v in props.items()):
            continue
        results.append(node)
    return results


MemoryGraphSource.match_edges = _match_edges
MemoryGraphSource.match_nodes = _match_nodes
