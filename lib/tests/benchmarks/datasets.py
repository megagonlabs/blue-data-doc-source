"""Synthetic document corpus generation for benchmarking.

All generators are deterministic (fixed seed). Three document types:

- structured:   Markdown headers + numbered subsections + prose paragraphs.
                SectionChunker should detect headers and segment cleanly.
- unstructured: Dense prose, no headers anywhere.
                WindowChunker and SentenceChunker are designed for this.
- mixed:        Numbered sections alternating with plain prose blocks.
                Exercises the SectionChunker's numbered-section regex path.
"""
import random
from dataclasses import dataclass
from typing import List

# ── Vocabulary ────────────────────────────────────────────────────────────────

_NOUNS = [
    "customer", "policy", "coverage", "premium", "deductible", "claim",
    "benefit", "requirement", "payment", "term", "condition", "period",
    "renewal", "cancellation", "agreement", "provision", "liability",
    "beneficiary", "schedule", "amendment", "exclusion", "limitation",
    "obligation", "authority", "document", "procedure", "applicant",
    "holder", "insurer", "certificate", "declaration", "endorsement",
    "waiver", "notice", "regulation", "compliance", "assessment",
]

_VERBS = [
    "shall", "must", "may", "requires", "provides", "specifies",
    "determines", "establishes", "ensures", "maintains", "terminates",
    "renews", "cancels", "covers", "excludes", "applies", "constitutes",
    "represents", "warrants", "indemnifies", "reimburses", "designates",
]

_ADJECTIVES = [
    "applicable", "effective", "written", "specified", "required", "eligible",
    "valid", "current", "prior", "subsequent", "additional", "reasonable",
    "necessary", "appropriate", "financial", "legal", "general", "specific",
    "individual", "annual", "total", "material", "relevant", "binding",
]

_CONNECTORS = [
    "in accordance with", "pursuant to", "subject to", "as specified in",
    "in the event that", "provided that", "notwithstanding the foregoing",
    "in addition to", "except as otherwise provided", "with respect to",
    "for the purpose of", "as set forth in", "consistent with",
]

_SECTION_TITLES = [
    "General Provisions", "Definitions and Interpretations",
    "Eligibility Requirements", "Coverage Terms and Conditions",
    "Payment Obligations", "Claims Procedure",
    "Exclusions and Limitations", "Renewal and Cancellation",
    "Dispute Resolution", "Governing Law",
    "Notices and Communications", "Amendment Procedures",
    "Liability and Indemnification", "Effective Date and Term",
    "Schedule of Benefits", "Compliance Requirements",
    "Regulatory Provisions", "Administrative Procedures",
    "Record Keeping Requirements", "Termination Conditions",
    "Subrogation Rights", "Assignment Restrictions",
    "Representations and Warranties", "Force Majeure",
]


# ── Text primitives ──────────────────────────────────────────────────────────

def _sentence(rng: random.Random) -> str:
    n = rng.choice(_NOUNS).capitalize()
    v = rng.choice(_VERBS)
    adj = rng.choice(_ADJECTIVES)
    obj = rng.choice(_NOUNS)
    conn = rng.choice(_CONNECTORS)
    extra = rng.choice(_NOUNS)
    templates = [
        f"The {n} {v} {adj} {obj} {conn} this agreement.",
        f"Any {adj} {obj} {conn} the {extra} shall be documented in writing.",
        f"The {extra} {v} comply with all {adj} requirements governing the {obj}.",
        f"{conn.capitalize()} the {n}, the {extra} {v} meet all {adj} standards.",
        f"All {adj} {obj} provisions apply to the {extra} unless otherwise specified.",
        f"The {n} {v} notify the {extra} of any {adj} change to the {obj}.",
        f"Written {adj} consent {v} be obtained prior to any modification of the {obj}.",
    ]
    return rng.choice(templates)


def _paragraph(rng: random.Random, n_sentences: int = None) -> str:
    n = n_sentences or rng.randint(3, 6)
    return " ".join(_sentence(rng) for _ in range(n))


def _word_count(text: str) -> int:
    return len(text.split())


# ── Document generators ──────────────────────────────────────────────────────

def generate_structured(target_words: int, rng: random.Random = None) -> str:
    """Markdown headers and numbered subsections — rich header signal for SectionChunker."""
    rng = rng or random.Random(42)
    lines: List[str] = []
    words = 0
    sec = 0

    while words < target_words:
        title = _SECTION_TITLES[sec % len(_SECTION_TITLES)]
        cycle = sec // len(_SECTION_TITLES)
        header = f"# {title}" if cycle == 0 else f"# {title} (Part {cycle + 1})"
        lines += [header, ""]

        for sub in range(1, rng.randint(2, 4)):
            sub_noun = rng.choice(_NOUNS).title()
            lines += [f"## {sec + 1}.{sub} {sub_noun} Requirements", ""]

            for _ in range(rng.randint(2, 4)):
                para = _paragraph(rng)
                lines += [para, ""]
                words += _word_count(para)
                if words >= target_words:
                    break
            if words >= target_words:
                break
        sec += 1

    return "\n".join(lines)


def generate_unstructured(target_words: int, rng: random.Random = None) -> str:
    """Dense prose, no headers. WindowChunker and SentenceChunker territory."""
    rng = rng or random.Random(42)
    paras: List[str] = []
    words = 0

    while words < target_words:
        para = _paragraph(rng, n_sentences=rng.randint(4, 8))
        paras.append(para)
        words += _word_count(para)

    return "\n\n".join(paras)


def generate_mixed(target_words: int, rng: random.Random = None) -> str:
    """Numbered section headers mixed with plain prose blocks."""
    rng = rng or random.Random(42)
    lines: List[str] = []
    words = 0
    titles = _SECTION_TITLES[:]
    rng.shuffle(titles)
    title_cycle = 0

    while words < target_words:
        if rng.random() > 0.35:
            t = titles[title_cycle % len(titles)]
            cycle = title_cycle // len(titles)
            heading = f"{title_cycle + 1}. {t}" if cycle == 0 else f"{title_cycle + 1}. {t} (continued)"
            lines += [heading, ""]
            title_cycle += 1

        for _ in range(rng.randint(2, 5)):
            para = _paragraph(rng)
            lines += [para, ""]
            words += _word_count(para)
            if words >= target_words:
                break

    return "\n".join(lines)


# ── Dataset and corpus builders ──────────────────────────────────────────────

@dataclass
class Dataset:
    name: str
    doc_type: str
    text: str
    word_count: int

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {self.word_count:,} words)"


SMALL  = 2_000
MEDIUM = 10_000
LARGE  = 50_000


def build_corpus(target_words: int = MEDIUM) -> List[Dataset]:
    """One dataset per doc type at the requested word count."""
    return [_make(target_words, t) for t in ("structured", "unstructured", "mixed")]


def _make(target_words: int, doc_type: str) -> Dataset:
    rng = random.Random(42)
    generators = {
        "structured":   generate_structured,
        "unstructured": generate_unstructured,
        "mixed":        generate_mixed,
    }
    text = generators[doc_type](target_words, rng)
    return Dataset(
        name=f"{doc_type}_{target_words}w",
        doc_type=doc_type,
        text=text,
        word_count=_word_count(text),
    )
