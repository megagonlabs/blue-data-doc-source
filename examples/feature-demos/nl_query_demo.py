"""
NLQueryParser Demo
==================
Shows how NLQueryParser converts natural-language strings into structured
DOCUMENT.* op dicts that DocumentDataSource.execute_query accepts.

NLQueryParser uses OpenAI (gpt-4o-mini) by default to parse NL.
It also accepts a custom parse_fn — useful for testing or alternative backends.

This demo uses a simple regex parse_fn so it runs without an API key.
The OpenAI-backed version is identical in interface; only parse quality differs.

Op outputs:
  search          → {"op": "DOCUMENT.SEARCH",          "query": "…", "k": …}
  find            → {"op": "DOCUMENT.FIND",             "filter": {…}}
  find_then_search→ {"op": "DOCUMENT.FIND_THEN_SEARCH", "find_filter": {…}, "query": "…"}
  ingest          → {"op": "DOCUMENT.INGEST",           "path": "…"}
  delete          → {"op": "DOCUMENT.DELETE",           "path": "…"}

Run:
    python3 examples/feature-demos/nl_query_demo.py

Run with OpenAI (higher parse quality):
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/nl_query_demo.py --openai
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_memory_store import LocalMemoryVectorStore

from blue_data_doc_source import DocumentDataSource, DummyEmbeddingModel, NLQueryParser

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
COLLECTION  = "nl_demo"
DIV  = "=" * 68
DIV2 = "-" * 68

SEARCH_INPUTS = [
    "find information about the deductible",
    "search for emergency room coverage",
    "what is the out-of-pocket maximum?",
    "show me prescription drug tiers",
    "look up pre-existing condition exclusions",
    "retrieve documents about enrollment periods",
]

FIND_INPUTS = [
    "list all contracts",
    "show me invoices",
    "which documents are policies?",
    "find all research papers",
]

FIND_THEN_SEARCH_INPUTS = [
    ("find the termination clause in our contracts",
     "contract", "termination notice period"),
    ("what does the policy say about deductibles",
     "policy", "deductible out-of-pocket"),
    ("show me payment terms in invoices",
     "invoice", "payment due terms"),
]

INGEST_INPUTS = [
    "ingest report.pdf",
    "add document /docs/policy_2025.txt",
    "load file contracts/agreement.pdf",
    "import /data/handbook.docx",
]

DELETE_INPUTS = [
    "delete report.pdf",
    "remove document /docs/old_policy.txt",
]


# ── simple regex parse_fn (no OpenAI needed) ──────────────────────────────

_INGEST_RE = re.compile(
    r"\b(?:ingest|add|load|import|process)\b.*?([^\s]+\.(?:pdf|txt|docx|doc|md))",
    re.I,
)
_DELETE_RE = re.compile(
    r"\b(?:delete|remove|drop)\b.*?([^\s]+\.(?:pdf|txt|docx|doc|md))",
    re.I,
)
_FIND_RE = re.compile(
    r"\b(?:list|show me|find all|which documents are)\b.+?\b"
    r"(contracts?|invoices?|polic(?:y|ies)|research papers?)\b",
    re.I,
)
_TYPE_MAP = {
    "contract": "contract", "contracts": "contract",
    "invoice": "invoice", "invoices": "invoice",
    "policy": "policy", "policies": "policy",
    "research paper": "research_paper", "research papers": "research_paper",
}
# "find [query] in [our] [doc_type]" / "show me [query] in [doc_type]"
_FTS_RE_B = re.compile(
    r"\b(?:find|show\s+me|get)\b\s+(.+?)\s+\bin\b\s+(?:\w+\s+)?"
    r"(contracts?|invoices?|polic(?:y|ies))\b",
    re.I,
)
# "what does [doc_type] say about [query]" / "find [doc_type] about [query]"
_FTS_RE_A = re.compile(
    r"\b(?:what\s+does|find)\b.+?\b"
    r"(contracts?|invoices?|polic(?:y|ies))\b.+?\b(?:about|say\s+about|for)\b(.+)",
    re.I,
)


def _regex_parse(text: str) -> dict:
    m = _DELETE_RE.search(text)
    if m:
        return {"op": "DOCUMENT.DELETE", "path": m.group(1)}
    m = _INGEST_RE.search(text)
    if m:
        return {"op": "DOCUMENT.INGEST", "path": m.group(1)}
    m = _FTS_RE_B.search(text)
    if m:
        query    = m.group(1).strip()
        raw_type = m.group(2).lower()
        doc_type = _TYPE_MAP.get(raw_type, raw_type.rstrip("s"))
        return {
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": doc_type},
            "query": query,
            "k": 5,
        }
    m = _FTS_RE_A.search(text)
    if m:
        raw_type = m.group(1).lower()
        doc_type = _TYPE_MAP.get(raw_type, raw_type.rstrip("s"))
        query    = m.group(2).strip()
        return {
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": doc_type},
            "query": query,
            "k": 5,
        }
    m = _FIND_RE.search(text)
    if m:
        raw_type = m.group(1).lower()
        doc_type = _TYPE_MAP.get(raw_type, raw_type.rstrip("s"))
        return {"op": "DOCUMENT.FIND", "filter": {"document_type": doc_type}}
    return {"op": "DOCUMENT.SEARCH", "query": text, "k": 5}


def fmt_op(op: dict) -> str:
    parts = [f"op={op.get('op')!r}"]
    for k, v in op.items():
        if k != "op":
            parts.append(f"{k}={v!r}")
    return "{" + ", ".join(parts) + "}"


def run(use_openai: bool) -> None:
    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: --openai requires OPENAI_API_KEY to be set.")
            sys.exit(1)
        parser = NLQueryParser(api_key=api_key)
        backend_note = "OpenAI gpt-4o-mini"
    else:
        parser = NLQueryParser(parse_fn=_regex_parse)
        backend_note = "regex parse_fn (no API key)"

    print()
    print(DIV)
    print("NLQueryParser Demo")
    print(DIV)
    print()
    print(f"  Backend: {backend_note}")
    print()
    print("  NLQueryParser maps natural-language strings to structured op dicts.")
    print("  Default backend: OpenAI gpt-4o-mini.")
    print("  Custom parse_fn: inject any callable (text: str) -> dict.")
    print()

    # ── Search intent ──────────────────────────────────────────────────────
    print(DIV2)
    print("Search intent  →  DOCUMENT.SEARCH")
    print(DIV2)
    print()
    for text in SEARCH_INPUTS:
        op = parser.parse(text)
        print(f"  Input : {text!r}")
        print(f"  Op    : {fmt_op(op)}")
        print()

    # ── Find intent ────────────────────────────────────────────────────────
    print(DIV2)
    print("Find intent  →  DOCUMENT.FIND")
    print(DIV2)
    print()
    print("  'List all contracts' → filter DocumentRecords by type, no embedding.")
    print()
    for text in FIND_INPUTS:
        op = parser.parse(text)
        print(f"  Input : {text!r}")
        print(f"  Op    : {fmt_op(op)}")
        print()

    # ── Find-then-search intent ────────────────────────────────────────────
    print(DIV2)
    print("Find-then-search intent  →  DOCUMENT.FIND_THEN_SEARCH")
    print(DIV2)
    print()
    print("  Two-part questions: identify document type AND retrieve specific text.")
    print("  NLQueryParser emits FIND_THEN_SEARCH when it detects both intents.")
    print()
    for text, _doc_type, _query in FIND_THEN_SEARCH_INPUTS:
        op = parser.parse(text)
        print(f"  Input : {text!r}")
        print(f"  Op    : {fmt_op(op)}")
        print()

    # ── Ingest intent ──────────────────────────────────────────────────────
    print(DIV2)
    print("Ingest intent  →  DOCUMENT.INGEST")
    print(DIV2)
    print()
    for text in INGEST_INPUTS:
        op = parser.parse(text)
        print(f"  Input : {text!r}")
        print(f"  Op    : {fmt_op(op)}")
        print()

    # ── Delete intent ──────────────────────────────────────────────────────
    print(DIV2)
    print("Delete intent  →  DOCUMENT.DELETE")
    print(DIV2)
    print()
    for text in DELETE_INPUTS:
        op = parser.parse(text)
        print(f"  Input : {text!r}")
        print(f"  Op    : {fmt_op(op)}")
        print()

    # ── Live end-to-end ────────────────────────────────────────────────────
    print(DIV2)
    print("Live end-to-end  —  execute_query_nl on a real document")
    print(DIV2)
    print()

    source = DocumentDataSource(
        name="nl_demo",
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=LocalMemoryVectorStore(),
        chunk_size=200,
        overlap=30,
        default_collection=COLLECTION,
        nl_parser=parser,
    )
    source.execute_query({"op": "DOCUMENT.ENSURE_INDEX", "collection": COLLECTION})
    source.execute_query({
        "op": "DOCUMENT.INGEST",
        "path": str(DEFAULT_DOC),
        "collection": COLLECTION,
    })
    print(f"  Ingested: {DEFAULT_DOC.name}")
    print()

    queries = [
        "find information about the deductible",
        "show me who is eligible for coverage",
        "search for prescription drug costs",
    ]
    for nl in queries:
        print(f"  execute_query_nl({nl!r})")
        results = source.execute_query_nl(nl, collection=COLLECTION)
        if results:
            r       = results[0]
            snippet = r.get("text", "")[:100].replace("\n", " ")
            section = f"  [{r['section_title']}]" if r.get("section_title") else ""
            print(f"    → top result{section}: {snippet}…")
        else:
            print("    → no results")
        print()

    print(DIV)
    print("How to wire it up")
    print(DIV)
    print()
    print("  # Default: uses OpenAI gpt-4o-mini")
    print("  parser = NLQueryParser(api_key='sk-...')")
    print()
    print("  # Custom: inject any callable")
    print("  parser = NLQueryParser(parse_fn=lambda t: {'op': 'DOCUMENT.SEARCH', 'query': t})")
    print()
    print("  source = DocumentDataSource(")
    print("      name='docs',")
    print("      nl_parser=parser,   # ← inject at construction time")
    print("      …,")
    print("  )")
    print("  results = source.execute_query_nl('find the termination clause')")
    print()
    print("  Without nl_parser=, execute_query_nl raises ValueError.")
    print()


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser(description="NLQueryParser demo")
    parser_.add_argument("--openai", action="store_true",
                         help="Use OpenAI backend (requires OPENAI_API_KEY)")
    args = parser_.parse_args()
    run(args.openai)
