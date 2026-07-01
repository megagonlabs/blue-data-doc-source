import pytest
from blue_doc_source.source.document.chunkers import WindowChunker


def test_basic_chunking():
    chunker = WindowChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk("a b c d e f g")
    # stride=2, so windows start at 0,2,4,6 → "a b c", "c d e", "e f g", "g"
    assert chunks == ["a b c", "c d e", "e f g", "g"]


def test_single_chunk_when_text_fits():
    chunker = WindowChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("one two three")
    assert chunks == ["one two three"]


def test_overlap_carries_words():
    chunker = WindowChunker(chunk_size=4, overlap=2)
    chunks = chunker.chunk("a b c d e f")
    assert chunks[0].endswith(chunks[1].split()[0]) or chunks[1].startswith(chunks[0].split()[-2])


def test_empty_string_returns_empty():
    chunker = WindowChunker(chunk_size=5, overlap=1)
    assert chunker.chunk("") == []


def test_whitespace_only_returns_empty():
    chunker = WindowChunker(chunk_size=5, overlap=1)
    assert chunker.chunk("   \n\t  ") == []


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        WindowChunker(chunk_size=0, overlap=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        WindowChunker(chunk_size=5, overlap=-1)


def test_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError):
        WindowChunker(chunk_size=5, overlap=5)


def test_chunk_with_titles_all_none():
    chunker = WindowChunker(chunk_size=3, overlap=1)
    results = chunker.chunk_with_titles("a b c d e f g")
    assert all(title is None for title, _ in results)
    assert [text for _, text in results] == chunker.chunk("a b c d e f g")
