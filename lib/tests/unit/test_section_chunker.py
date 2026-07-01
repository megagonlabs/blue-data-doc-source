import pytest
from blue_doc_source.source.document.chunkers.section_chunker import SectionChunker


def test_plain_text_falls_back_to_window():
    text = " ".join(["word"] * 20)
    chunker = SectionChunker(chunk_size=5, overlap=1)
    results = chunker.chunk_with_titles(text)
    assert len(results) > 0
    assert all(title is None for title, _ in results)


def test_markdown_headers_detected():
    text = "# Introduction\nThis is the intro.\n## Methods\nHere are the methods."
    chunker = SectionChunker(chunk_size=200, overlap=0)
    results = chunker.chunk_with_titles(text)
    titles = [t for t, _ in results]
    assert "Introduction" in titles
    assert "Methods" in titles


def test_numbered_section_detected():
    text = "1. Overview\nThis is the overview.\n2. Details\nHere are details."
    chunker = SectionChunker(chunk_size=200, overlap=0)
    results = chunker.chunk_with_titles(text)
    titles = [t for t, _ in results]
    assert any("Overview" in (t or "") for t in titles)
    assert any("Details" in (t or "") for t in titles)


def test_caps_header_detected():
    text = "INTRODUCTION\nThis is the intro text.\nMETHODS\nHere are the methods."
    chunker = SectionChunker(chunk_size=200, overlap=0)
    results = chunker.chunk_with_titles(text)
    titles = [t for t, _ in results]
    assert "INTRODUCTION" in titles
    assert "METHODS" in titles


def test_underlined_header_detected():
    text = "Overview\n========\nThis is the overview.\nDetails\n-------\nHere are details."
    chunker = SectionChunker(chunk_size=200, overlap=0)
    results = chunker.chunk_with_titles(text)
    titles = [t for t, _ in results]
    assert "Overview" in titles
    assert "Details" in titles


def test_section_title_propagates_across_window_chunks():
    content = " ".join(["word"] * 30)
    text = f"# BigSection\n{content}"
    chunker = SectionChunker(chunk_size=5, overlap=1)
    results = chunker.chunk_with_titles(text)
    assert len(results) > 1
    assert all(title == "BigSection" for title, _ in results)


def test_empty_text_returns_empty():
    chunker = SectionChunker(chunk_size=200, overlap=0)
    assert chunker.chunk_with_titles("") == []


def test_chunk_text_not_empty():
    text = "# Intro\nSome content here.\n# Methods\nMore content."
    chunker = SectionChunker(chunk_size=200, overlap=0)
    results = chunker.chunk_with_titles(text)
    assert all(chunk_text.strip() for _, chunk_text in results)
