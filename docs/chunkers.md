# Chunkers

Chunking splits document text into overlapping segments that fit within an embedding model's context window, while preserving enough surrounding context to make each chunk independently useful for retrieval.

All chunkers implement the `Chunker` ABC:

```python
class Chunker(ABC):
    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        ...
```

The return value is a list of `(section_title, chunk_text)` pairs. `section_title` is `None` when the chunk has no detected parent section heading.

---

## SectionChunker (default)

`SectionChunker` is the default chunker. It detects section headers in the text, splits at each boundary, then applies word-count windowing within each section. The detected heading becomes `section_title` on every chunk that falls under it.

**When to use:** Structured documents — policy PDFs, Word documents, markdown files, technical specs. Any file where sections carry meaning and agents will need to cite "which part of the document" answered a query.

**Header patterns detected:**

| Pattern | Example |
|---|---|
| Markdown headers | `# Title`, `## Section`, `### Sub` |
| Numbered sections | `1. Intro`, `2.3 Background`, `4.1.2 Detail` |
| ALL-CAPS headings | `ELIGIBILITY REQUIREMENTS` (3–8 words) |
| Underlined headings | `Title` followed by `===` or `---` on the next line |

When no headers are found, `SectionChunker` falls back to plain window chunking and returns `section_title=None` on all chunks — identical to `WindowChunker`.

```python
from blue_doc_source import SectionChunker

chunker = SectionChunker(chunk_size=1000, overlap=150)
chunks = chunker.chunk_with_titles(text)
# [("Eligibility Requirements", "Employees must be..."), ...]
```

---

## WindowChunker

`WindowChunker` is a word-count sliding window with no section awareness. Every chunk gets `section_title=None`.

**When to use:** Continuous prose with no header structure — transcripts, emails, narratives, raw scraped text. Use it directly when you know the input is unstructured and section titles would be noise.

Chunk boundaries are word-aligned, so `chunk_size` is exact in words but variable in characters.

```python
from blue_doc_source import WindowChunker

chunker = WindowChunker(chunk_size=200, overlap=40)
chunks = chunker.chunk_with_titles(text)
# [(None, "The quick brown fox..."), ...]
```

---

## SentenceChunker

`SentenceChunker` closes chunks at sentence boundaries — no sentence is ever split across two chunks. It accumulates sentences until the next sentence would push the chunk past `chunk_size` words, then closes at the last complete sentence.

Overlap is built by carrying trailing sentences from the previous chunk until the overlap word budget is filled, so overlap boundaries are also sentence-aligned.

**When to use:** When downstream agents reason over or display individual chunks and a mid-sentence break would degrade the quality of the result. Also useful for chunks fed to a re-ranker or summariser where sentence completeness matters.

**Sentence detection:** Uses `nltk.sent_tokenize` when available (install the `[nlp]` extra and run `nltk.download('punkt_tab')`). Falls back to a regex splitter on `.`, `!`, `?` boundaries when nltk is absent. When no sentence boundaries are detected at all (e.g. raw tokens without punctuation), degrades gracefully to word-count splitting.

```python
from blue_doc_source import SentenceChunker

chunker = SentenceChunker(chunk_size=300, overlap=50)
chunks = chunker.chunk_with_titles(text)
```

---

## Choosing a chunker

| Situation | Recommended chunker |
|---|---|
| Document has detectable headings | `SectionChunker` (default) |
| Unstructured prose, no headings | `WindowChunker` |
| Agent reads or displays individual chunks | `SentenceChunker` |
| Feeding chunks to a summariser or re-ranker | `SentenceChunker` |
| Maximum control over exact word count | `WindowChunker` |

---

## Tuning chunk_size and overlap

All chunkers measure size in **words**, not characters or tokens.

**chunk_size** determines the upper bound on how much text each chunk contains. A smaller value produces more chunks, each with narrower focus — better for precise retrieval at the cost of more embeddings and more vector store entries. A larger value produces fewer chunks with broader context — better for summarisation but may dilute relevance scores.

Rough starting points:

| Use case | chunk_size | overlap |
|---|---|---|
| Precise fact retrieval | 200–400 | 40–80 |
| General Q&A over documents | 500–800 | 100–150 |
| Broad context for summarisation | 800–1500 | 150–300 |

**overlap** controls how many words are shared between consecutive chunks. Overlap ensures that information near a chunk boundary appears in at least two chunks, reducing the chance that a relevant sentence falls entirely in a gap. Set overlap to roughly 10–20% of `chunk_size`. Overlap must be smaller than `chunk_size`.

---

## Injecting a custom chunker

Pass any `Chunker` subclass as `chunker=` on `DocumentPipeline`:

```python
from blue_doc_source import DocumentPipeline, SentenceChunker

pipeline = DocumentPipeline(
    chunker=SentenceChunker(chunk_size=300, overlap=60),
    embedding_model=...,
)
```

---

## Writing a custom chunker

Subclass `Chunker` and implement `chunk_with_titles`:

```python
from blue_doc_source import Chunker
from typing import List, Optional, Tuple

class ParagraphChunker(Chunker):
    """One chunk per paragraph, no overlap."""

    def chunk_with_titles(self, text: str) -> List[Tuple[Optional[str], str]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [(None, p) for p in paragraphs]
```

The pipeline calls `chunk_with_titles` on each page/document string. The returned `section_title` value propagates directly to `ChunkRecord.section_title`. Return `None` as the title when no section context applies.
