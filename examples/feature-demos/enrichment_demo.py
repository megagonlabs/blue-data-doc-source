"""
BasicEnricher Reference
=======================
Shows what BasicEnricher extracts from document chunks — no ML, no API key.

BasicEnricher is a pure-regex enricher that runs on every chunk during
ingestion. It produces two output groups:

  entities      — typed spans found in the text (field: "type", not "label")
    DATE        date-like strings (2024-01-01, January 1 2024, …)
    EMAIL       email addresses
    PERCENT     numeric percentages  (20%, 100%, …)
    MONEY       currency amounts     ($10, €35, £70, …)
    PROPER_NOUN multi-word capitalised phrases

  semantic_hints — scalar signals about the chunk's content
    word_count      total words
    sentence_count  approximate sentence count
    has_questions   True when "?" is present
    has_numbers     True when digits are present
    keywords        top-k content words (stopword-filtered, min 3 chars)

Run:
    python3 examples/feature-demos/enrichment_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from blue_doc_source import BasicEnricher, SectionChunker

DATA_DIR    = Path(__file__).parent.parent.parent / "data"
DEFAULT_DOC = DATA_DIR / "sample_policy.txt"
DIV  = "=" * 68
DIV2 = "-" * 68

SAMPLE_SNIPPETS = [
    (
        "Financial terms",
        "The deductible is $1,500 per individual ($3,000 per family) per calendar "
        "year. After the deductible is met, the plan covers 80% of in-network costs "
        "until the out-of-pocket maximum of $6,000 is reached. Emergency room visits "
        "carry a $300 copayment. Mail-order prescriptions are billed at 2.5× the "
        "30-day copay for a 90-day supply.",
    ),
    (
        "Dates and eligibility",
        "Open enrollment runs from November 1 to December 15 each year, with coverage "
        "beginning on January 1 of the following year. Contact us at "
        "enroll@blueshield.example.com by December 15, 2024. Are you eligible "
        "if you turn 26 during the plan year?",
    ),
    (
        "Proper nouns",
        "Blue Shield Health Insurance offers coverage in partnership with the "
        "American Medical Association and the National Institutes of Health. "
        "PolicyHolder Jane Doe is enrolled in the Blue Shield Premium Plan "
        "administered by National Healthcare Administrators.",
    ),
]


def show_enrichment(label: str, text: str, enricher: BasicEnricher) -> None:
    result = enricher.enrich(text)
    entities      = result.get("entities", [])
    hints         = result.get("semantic_hints", {})

    print()
    print(f"  ── {label} ──")
    print()
    print(f"  Text ({len(text.split())} words):")
    print(f"    {text[:120].replace(chr(10), ' ')}…")
    print()

    if entities:
        print("  Entities:")
        by_type: dict[str, list[str]] = {}
        for e in entities:
            by_type.setdefault(e["type"], []).append(e["text"])
        for etype, values in sorted(by_type.items()):
            print(f"    {etype:<14}  {values}")
    else:
        print("  Entities: (none detected)")

    print()
    print("  Semantic hints:")
    for key in ["word_count", "sentence_count", "has_questions", "has_numbers"]:
        print(f"    {key:<20}  {hints.get(key)}")
    kw = hints.get("keywords", [])
    print(f"    {'keywords':<20}  {kw[:8]}")


def run() -> None:
    enricher = BasicEnricher(top_k_keywords=10)

    print()
    print(DIV)
    print("BasicEnricher Reference")
    print(DIV)
    print()
    print("  Pure regex. No ML. No API key. Runs on every chunk during ingestion.")
    print()

    # ── Synthetic snippets ─────────────────────────────────────────────────
    print(DIV2)
    print("Synthetic snippets  (entity types shown per sample)")
    print(DIV2)
    for label, text in SAMPLE_SNIPPETS:
        show_enrichment(label, text, enricher)

    # ── Live document ──────────────────────────────────────────────────────
    print()
    print(DIV2)
    print("Live document  (sample_policy.txt — first 4 chunks via SectionChunker)")
    print(DIV2)

    text   = DEFAULT_DOC.read_text(encoding="utf-8")
    chunks = SectionChunker(chunk_size=200, overlap=30).chunk_with_titles(text)

    for i, (title, chunk_text) in enumerate(chunks[:4]):
        result  = enricher.enrich(chunk_text)
        entities = result.get("entities", [])
        hints    = result.get("semantic_hints", {})
        n_ent    = len(entities)
        kw       = hints.get("keywords", [])[:5]
        title_s  = f"  [{title}]" if title else ""
        print()
        print(f"  chunk {i:02d}{title_s}")
        print(f"    words={hints.get('word_count')}  sentences={hints.get('sentence_count')}  "
              f"has_numbers={hints.get('has_numbers')}  entities={n_ent}")
        print(f"    keywords : {kw}")
        if entities:
            by_type: dict[str, list[str]] = {}
            for e in entities:
                by_type.setdefault(e["type"], []).append(e["text"])
            for etype, vals in sorted(by_type.items()):
                print(f"    {etype:<14}: {vals[:4]}")

    # ── Integration note ───────────────────────────────────────────────────
    print()
    print(DIV)
    print("Integration")
    print(DIV)
    print()
    print("  BasicEnricher runs automatically inside DocumentPipeline:")
    print()
    print("    pipeline = DocumentPipeline(enricher=BasicEnricher())")
    print("    records  = pipeline.ingest('doc.txt')")
    print("    chunk    = records[0]")
    print("    chunk['entities']       # → list of {label, text} dicts")
    print("    chunk['semantic_hints'] # → {word_count, keywords, …}")
    print()
    print("  Swap in a custom Enricher subclass for ML-powered entity recognition:")
    print("    pipeline = DocumentPipeline(enricher=SpacyEnricher())")
    print()


if __name__ == "__main__":
    run()
