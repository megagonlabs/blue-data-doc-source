import pytest
from docx import Document as DocxDocument

from blue_doc_source.source.document.parsers.docx_parser import DocxParser


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_docx(tmp_path, specs, filename="test.docx"):
    """Create a real .docx file from a list of paragraph specs.

    Each spec is either:
    - str: a Normal paragraph
    - {"text": str, "style": str}: a paragraph with the given Word style
    - {"table": [[cell, ...], ...]}: a table (rows × cells)
    """
    doc = DocxDocument()
    for spec in specs:
        if isinstance(spec, str):
            doc.add_paragraph(spec)
        elif "table" in spec:
            rows = spec["table"]
            tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    tbl.rows[r_idx].cells[c_idx].text = cell_text
        else:
            doc.add_paragraph(spec["text"], style=spec.get("style", "Normal"))
    path = tmp_path / filename
    doc.save(str(path))
    return str(path)


# ── Basic extraction ──────────────────────────────────────────────────────────

def test_parse_plain_paragraphs(tmp_path):
    path = make_docx(tmp_path, ["First paragraph.", "Second paragraph."])
    result = DocxParser().parse(path)
    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_parse_heading1_becomes_h1(tmp_path):
    path = make_docx(tmp_path, [{"text": "Introduction", "style": "Heading 1"}])
    result = DocxParser().parse(path)
    assert "# Introduction" in result


def test_parse_heading2_becomes_h2(tmp_path):
    path = make_docx(tmp_path, [{"text": "Background", "style": "Heading 2"}])
    result = DocxParser().parse(path)
    assert "## Background" in result


def test_parse_heading3_becomes_h3(tmp_path):
    path = make_docx(tmp_path, [{"text": "Details", "style": "Heading 3"}])
    result = DocxParser().parse(path)
    assert "### Details" in result


def test_parse_normal_paragraph_has_no_hash_prefix(tmp_path):
    path = make_docx(tmp_path, ["Just a normal sentence."])
    result = DocxParser().parse(path)
    assert result.strip() == "Just a normal sentence."


def test_parse_skips_empty_paragraphs(tmp_path):
    path = make_docx(tmp_path, ["First.", "", "Second."])
    result = DocxParser().parse(path)
    lines = [l for l in result.splitlines() if l.strip()]
    assert len(lines) == 2


def test_parse_preserves_document_order(tmp_path):
    path = make_docx(tmp_path, [
        {"text": "Section One", "style": "Heading 1"},
        "Section one content.",
        {"text": "Section Two", "style": "Heading 1"},
        "Section two content.",
    ])
    result = DocxParser().parse(path)
    pos_h1 = result.index("# Section One")
    pos_p1 = result.index("Section one content.")
    pos_h2 = result.index("# Section Two")
    pos_p2 = result.index("Section two content.")
    assert pos_h1 < pos_p1 < pos_h2 < pos_p2


# ── Table extraction ──────────────────────────────────────────────────────────

def test_parse_table_renders_as_pipe_delimited(tmp_path):
    path = make_docx(tmp_path, [
        {"table": [["Name", "Amount"], ["Alice", "100"], ["Bob", "200"]]}
    ])
    result = DocxParser().parse(path)
    assert "| Name | Amount |" in result
    assert "| Alice | 100 |" in result
    assert "| Bob | 200 |" in result


def test_parse_table_in_document_order(tmp_path):
    path = make_docx(tmp_path, [
        {"text": "Summary", "style": "Heading 1"},
        {"table": [["Col A", "Col B"], ["1", "2"]]},
        "Closing paragraph.",
    ])
    result = DocxParser().parse(path)
    pos_heading = result.index("# Summary")
    pos_table   = result.index("| Col A")
    pos_para    = result.index("Closing paragraph.")
    assert pos_heading < pos_table < pos_para


# ── Structured document ───────────────────────────────────────────────────────

def test_parse_full_structured_document(tmp_path):
    path = make_docx(tmp_path, [
        {"text": "Policy Overview", "style": "Heading 1"},
        "This policy covers all employees.",
        {"text": "Eligibility", "style": "Heading 2"},
        "Employees must be full-time to qualify.",
        {"text": "Payment Schedule", "style": "Heading 2"},
        {"table": [["Month", "Amount"], ["January", "$500"]]},
    ])
    result = DocxParser().parse(path)
    assert "# Policy Overview" in result
    assert "## Eligibility" in result
    assert "## Payment Schedule" in result
    assert "| Month | Amount |" in result


def test_parse_section_titles_detectable_by_section_chunker(tmp_path):
    from blue_doc_source.source.document.chunkers import SectionChunker
    path = make_docx(tmp_path, [
        {"text": "Eligibility Requirements", "style": "Heading 1"},
        "Customers must be 18 or older and a legal resident.",
        {"text": "Payment Terms", "style": "Heading 1"},
        "Payment is due within 30 days of invoice.",
    ])
    text = DocxParser().parse(path)
    chunks = SectionChunker(chunk_size=200, overlap=0).chunk_with_titles(text)
    titles = {title for title, _ in chunks if title}
    assert "Eligibility Requirements" in titles
    assert "Payment Terms" in titles


# ── Error handling ────────────────────────────────────────────────────────────

def test_parse_file_not_found():
    with pytest.raises(FileNotFoundError):
        DocxParser().parse("/nonexistent/path/doc.docx")


def test_parse_wrong_extension(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content")
    with pytest.raises(ValueError, match=r"\.docx"):
        DocxParser().parse(str(f))


def test_parse_empty_document(tmp_path):
    path = make_docx(tmp_path, [])
    result = DocxParser().parse(path)
    assert result == ""
