import pytest
from unittest.mock import patch
from blue_data_doc_source.source.document.chunkers import SentenceChunker


SENTENCES = (
    "The deductible resets on January 1 of each plan year. "
    "Preventive care is covered at one hundred percent with no cost sharing. "
    "Out-of-pocket maximums apply separately to individual and family coverage. "
    "Claims must be submitted within ninety days of the service date. "
    "Prior authorization is required for all elective surgical procedures."
)


def test_no_sentence_split_mid_thought():
    chunker = SentenceChunker(chunk_size=15, overlap=5)
    chunks = chunker.chunk(SENTENCES)
    for chunk in chunks:
        for sent in SENTENCES.split(". "):
            sent = sent.strip().rstrip(".")
            words = sent.split()
            # if sentence starts appear in this chunk, the full sentence must be there
            if words[0] in chunk and words[1] in chunk:
                assert all(w in chunk for w in words), (
                    f"Sentence split mid-thought in chunk: {chunk!r}"
                )


def test_chunks_cover_all_text():
    chunker = SentenceChunker(chunk_size=15, overlap=0)
    chunks = chunker.chunk(SENTENCES)
    all_words = set(SENTENCES.split())
    recovered = set(w for c in chunks for w in c.split())
    assert all_words == recovered


def test_single_chunk_when_text_fits():
    chunker = SentenceChunker(chunk_size=200, overlap=10)
    chunks = chunker.chunk(SENTENCES)
    assert len(chunks) == 1
    assert chunks[0].strip() == SENTENCES.strip()


def test_overlap_carries_sentence_into_next_chunk():
    text = "First sentence here. Second sentence follows. Third sentence ends."
    chunker = SentenceChunker(chunk_size=6, overlap=3)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # A sentence from the first chunk should appear in the second
    first_chunk_sents = set(chunks[0].split(". "))
    second_chunk_words = chunks[1]
    assert any(s.split()[0] in second_chunk_words for s in first_chunk_sents if s)


def test_empty_string_returns_empty():
    assert SentenceChunker(chunk_size=10, overlap=2).chunk("") == []


def test_whitespace_only_returns_empty():
    assert SentenceChunker(chunk_size=10, overlap=2).chunk("   \n\t  ") == []


def test_falls_back_to_word_chunk_when_no_sentence_boundaries():
    chunker = SentenceChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk("a b c d e f g")
    # No punctuation — falls back to word-count splitting
    assert len(chunks) > 1
    assert chunks[0] == "a b c"


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=0, overlap=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=5, overlap=-1)


def test_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=5, overlap=5)


def test_chunk_with_titles_returns_none_titles():
    chunker = SentenceChunker(chunk_size=15, overlap=5)
    results = chunker.chunk_with_titles(SENTENCES)
    assert all(title is None for title, _ in results)
    assert [text for _, text in results] == chunker.chunk(SENTENCES)


def test_regex_fallback_used_when_nltk_unavailable():
    chunker = SentenceChunker(chunk_size=10, overlap=3)
    with patch.dict("sys.modules", {"nltk": None}):
        chunks = chunker.chunk(SENTENCES)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)
