import sys
import pytest
from unittest.mock import MagicMock, patch

from blue_data_doc_source.source.document.parsers.pdf_parser import PDFParser, _preprocess_image


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


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_returns_text(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Page one content.", "Page two content."])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 2
    assert result[0] == {"page_num": 1, "text": "Page one content."}
    assert result[1] == {"page_num": 2, "text": "Page two content."}


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_skips_blank_pages(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Content.", None, "More content."])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 2
    assert result[0]["page_num"] == 1
    assert result[1]["page_num"] == 3


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_skips_whitespace_only_pages(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Real content.", "   \n\t  "])

    result = PDFParser().parse_pages(str(pdf_path))

    assert len(result) == 1
    assert result[0]["page_num"] == 1


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_page_numbers_are_1based(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["A", "B", "C"])

    result = PDFParser().parse_pages(str(pdf_path))

    assert [r["page_num"] for r in result] == [1, 2, 3]


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
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


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_parse_pages_all_blank_returns_empty(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader([None, None])

    result = PDFParser().parse_pages(str(pdf_path))

    assert result == []


# --- OCR fallback tests ---

@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_ocr_fallback_called_for_blank_page(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader(["Normal text.", None])

    parser = PDFParser(ocr_fallback=True)
    ocr_return = {"text": "OCR extracted text", "ocr_confidence": 87.5}
    with patch.object(parser, "_ocr_page", return_value=ocr_return) as mock_ocr:
        result = parser.parse_pages(str(pdf_path))

    mock_ocr.assert_called_once_with(str(pdf_path), 2)
    assert len(result) == 2
    assert result[0] == {"page_num": 1, "text": "Normal text."}
    assert result[1] == {
        "page_num": 2,
        "text": "OCR extracted text",
        "ocr": True,
        "ocr_confidence": 87.5,
    }


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
def test_ocr_fallback_skips_page_when_ocr_returns_empty(mock_reader_cls, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    mock_reader_cls.return_value = _mock_reader([None])

    parser = PDFParser(ocr_fallback=True)
    with patch.object(parser, "_ocr_page", return_value={}):
        result = parser.parse_pages(str(pdf_path))

    assert result == []


@patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader")
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


# --- _ocr_page internals: confidence extraction and pre-processing ---

def _make_ocr_modules(words, confs):
    """Build mock pdf2image and pytesseract modules for sys.modules injection."""
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    mock_image.point.return_value = mock_image

    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_path.return_value = [mock_image]

    mock_output_cls = MagicMock()
    mock_pytesseract = MagicMock()
    mock_pytesseract.Output = mock_output_cls
    mock_pytesseract.image_to_data.return_value = {
        "text": words,
        "conf": [str(c) for c in confs],
    }

    mock_pil_enhance = MagicMock()
    mock_pil_enhance.Contrast.return_value.enhance.return_value = mock_image
    mock_pil = MagicMock()
    mock_pil.ImageEnhance = mock_pil_enhance

    return mock_pdf2image, mock_pytesseract, mock_pil, mock_image


def test_ocr_page_returns_text_and_confidence(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    mock_pdf2image, mock_pytesseract, mock_pil, _ = _make_ocr_modules(
        words=["Invoice", "Amount", "100.00", ""],
        confs=[85, 90, 80, -1],
    )
    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        result = parser._ocr_page(str(pdf_path), 1)

    assert result["text"] == "Invoice Amount 100.00"
    expected_conf = round((85 + 90 + 80) / 3, 1)
    assert result["ocr_confidence"] == expected_conf


def test_ocr_page_filters_negative_confidence_words(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    # conf=-1 marks non-word tokens (line/block separators in Tesseract output)
    mock_pdf2image, mock_pytesseract, mock_pil, _ = _make_ocr_modules(
        words=["Hello", "", "World", ""],
        confs=[92, -1, 88, -1],
    )
    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        result = parser._ocr_page(str(pdf_path), 1)

    assert result["text"] == "Hello World"
    assert result["ocr_confidence"] == round((92 + 88) / 2, 1)


def test_ocr_page_returns_empty_dict_when_no_confident_words(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    mock_pdf2image, mock_pytesseract, mock_pil, _ = _make_ocr_modules(
        words=["", ""],
        confs=[-1, -1],
    )
    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        result = parser._ocr_page(str(pdf_path), 1)

    assert result == {}


def test_ocr_page_returns_empty_dict_when_no_images(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_path.return_value = []  # no images extracted
    mock_pytesseract = MagicMock()
    mock_pil = MagicMock()

    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        result = parser._ocr_page(str(pdf_path), 1)

    assert result == {}
    mock_pytesseract.image_to_data.assert_not_called()


def test_ocr_page_uses_image_to_data_not_image_to_string(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    mock_pdf2image, mock_pytesseract, mock_pil, _ = _make_ocr_modules(
        words=["text"],
        confs=[80],
    )
    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        parser._ocr_page(str(pdf_path), 1)

    mock_pytesseract.image_to_data.assert_called_once()
    mock_pytesseract.image_to_string.assert_not_called()


def test_ocr_page_preprocesses_image_before_ocr(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    parser = PDFParser(ocr_fallback=True)

    mock_pdf2image, mock_pytesseract, mock_pil, mock_image = _make_ocr_modules(
        words=["text"],
        confs=[80],
    )
    with patch.dict(sys.modules, {
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract,
        "PIL": mock_pil,
        "PIL.ImageEnhance": mock_pil.ImageEnhance,
    }):
        parser._ocr_page(str(pdf_path), 1)

    # Preprocessing converts to grayscale, boosts contrast, binarises
    mock_image.convert.assert_called_once_with("L")
    mock_pil.ImageEnhance.Contrast.assert_called_once()
    mock_image.point.assert_called_once()


def test_ocr_confidence_present_in_parse_pages_result(mock_reader_cls=None, tmp_path=None):
    """ocr_confidence from _ocr_page flows through to the parse_pages dict."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        pdf_path = pathlib.Path(td) / "doc.pdf"
        pdf_path.write_bytes(b"fake")

        with patch("blue_data_doc_source.source.document.parsers.pdf_parser.PdfReader") as mock_cls:
            mock_cls.return_value = _mock_reader([None])
            parser = PDFParser(ocr_fallback=True)
            ocr_result = {"text": "Scanned invoice text", "ocr_confidence": 76.3}
            with patch.object(parser, "_ocr_page", return_value=ocr_result):
                result = parser.parse_pages(str(pdf_path))

    assert result[0]["ocr"] is True
    assert result[0]["ocr_confidence"] == 76.3
    assert result[0]["text"] == "Scanned invoice text"


# --- _preprocess_image unit test ---

def test_preprocess_image_converts_to_grayscale_and_binarises():
    mock_img = MagicMock()
    mock_grey = MagicMock()
    mock_img.convert.return_value = mock_grey
    mock_enhanced = MagicMock()
    mock_binarised = MagicMock()
    mock_enhanced.point.return_value = mock_binarised

    mock_enhance_cls = MagicMock()
    mock_enhance_cls.Contrast.return_value.enhance.return_value = mock_enhanced

    mock_pil = MagicMock()
    mock_pil.ImageEnhance = mock_enhance_cls

    with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.ImageEnhance": mock_enhance_cls}):
        result = _preprocess_image(mock_img)

    mock_img.convert.assert_called_once_with("L")
    mock_enhance_cls.Contrast.assert_called_once_with(mock_grey)
    mock_enhance_cls.Contrast.return_value.enhance.assert_called_once_with(2.0)
    mock_enhanced.point.assert_called_once()
    assert result is mock_binarised
