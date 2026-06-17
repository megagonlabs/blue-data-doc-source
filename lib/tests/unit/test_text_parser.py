import pytest
from blue_doc_source.source.document.parsers import TextParser


def test_parse_txt_file(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world", encoding="utf-8")
    assert TextParser().parse(str(f)) == "Hello world"


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        TextParser().parse("/nonexistent/path/file.txt")


def test_parse_non_txt_raises(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="TextParser only supports .txt"):
        TextParser().parse(str(f))
