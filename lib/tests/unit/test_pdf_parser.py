import pytest
from unittest.mock import MagicMock, patch

from blue_doc_source.source.document.parsers.pdf_parser import PDFParser


def _mock_reader(pages_text):
    """Build a mock PdfReader with the given list of page texts (None = blank)."""
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    reader = MagicMock()
    reader.pages = mock_pages
    return reader


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_returns_text(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Page one content.", "Page two content."])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 2
    assert result[0] == {"page_num": 1, "text": "Page one content."}
    assert result[1] == {"page_num": 2, "text": "Page two content."}


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_skips_blank_pages(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Content.", None, "More content."])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 2
    assert result[0]["page_num"] == 1
    assert result[1]["page_num"] == 3


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_skips_whitespace_only_pages(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Real content.", "   \n\t  "])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 1
    assert result[0]["page_num"] == 1


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_page_numbers_are_1based(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["A", "B", "C"])

    result = PDFParser().parse_pages(str(pdf_path))

    assert [r["page_num"] for r in result] == [1, 2, 3]


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_strips_whitespace(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["  trimmed  "])

    result = PDFParser().parse_pages(str(pdf_path))

    assert result[0]["text"] == "trimmed"


def test_parse_pages_file_not_found():
    with pytest.raises(FileNotFoundError):
        PDFParser().parse_pages("/nonexistent/path/doc.pdf")


def test_parse_pages_wrong_extension(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content")
    with pytest.raises(ValueError, match=r"\.pdf"):
        PDFParser().parse_pages(str(f))


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_all_blank_returns_empty(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader([None, None])

    result = PDFParser().parse_pages(str(pdf_path))

    assert result == []


# --- OCR fallback tests ---

@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_ocr_fallback_called_for_blank_page(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Normal text.", None])

    parser = PDFParser(ocr_fallback=True)
    with patch.object(parser, "_ocr_page", return_value="OCR extracted text") as mock_ocr:
        result = parser.parse_pages(str(pdf_path))

    mock_ocr.assert_called_once_with(str(pdf_path), 2)
    assert len(result) == 2
    assert result[0] == {"page_num": 1, "text": "Normal text."}
    assert result[1] == {"page_num": 2, "text": "OCR extracted text", "ocr": True}


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_ocr_fallback_skips_page_when_ocr_returns_empty(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader([None])

    parser = PDFParser(ocr_fallback=True)
    with patch.object(parser, "_ocr_page", return_value=""):
        result = parser.parse_pages(str(pdf_path))

    assert result == []


@patch("blue_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_ocr_fallback_not_called_when_disabled(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader([None])

    parser = PDFParser(ocr_fallback=False)
    with patch.object(parser, "_ocr_page") as mock_ocr:
        result = parser.parse_pages(str(pdf_path))

    mock_ocr.assert_not_called()
    assert result == []


def test_ocr_page_raises_import_error_when_deps_missing(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("pdf2image", "pytesseract"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pip install blue-data-doc-source\\[ocr\\]"):
            parser._ocr_page(str(pdf_path), 1)
