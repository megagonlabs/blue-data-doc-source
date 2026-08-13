import pytest
from unittest.mock import MagicMock

from blue_data_doc_source.source.document.nl_parser import NLQueryParser
from blue_data_doc_source.source.document.document_source import DocumentDataSource
from blue_data_doc_source.source.document.embeddings import DummyEmbeddingModel


def make_txt(tmp_path, content="The quick brown fox jumps over the lazy dog"):
    f = tmp_path / "test.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


def test_nl_parser_custom_parse_fn():
    parser = NLQueryParser(parse_fn=lambda text: {"op": "DOCUMENT.ENSURE_INDEX"})
    assert parser.parse("make sure the index exists") == {"op": "DOCUMENT.ENSURE_INDEX"}


def test_nl_parser_parse_fn_receives_text():
    received = []
    def capture(text):
        received.append(text)
        return {"op": "DOCUMENT.SEARCH", "query": text}

    parser = NLQueryParser(parse_fn=capture)
    parser.parse("find the termination clause")
    assert received == ["find the termination clause"]


def test_execute_query_nl_dispatches_search():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    parser = NLQueryParser(
        parse_fn=lambda text: {"op": "DOCUMENT.SEARCH", "query": "termination clause", "k": 3}
    )
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        nl_parser=parser,
    )
    results = source.execute_query_nl("find the termination clause")
    vector_source.knn.assert_called_once()
    assert results == []


def test_execute_query_nl_dispatches_ingest(tmp_path):
    path = make_txt(tmp_path)
    parser = NLQueryParser(parse_fn=lambda text: {"op": "DOCUMENT.INGEST", "path": path})
    source = DocumentDataSource(chunk_size=5, overlap=1, nl_parser=parser)
    records = source.execute_query_nl(f"ingest {path}")
    assert len(records) > 0


def test_execute_query_nl_without_parser_raises():
    source = DocumentDataSource()
    with pytest.raises(ValueError, match="nl_parser"):
        source.execute_query_nl("find something")


def test_execute_query_nl_passes_collection():
    vector_source = MagicMock()
    vector_source.knn.return_value = []
    parser = NLQueryParser(
        parse_fn=lambda text: {"op": "DOCUMENT.SEARCH", "query": "eligibility"}
    )
    source = DocumentDataSource(
        embedding_model=DummyEmbeddingModel(dim=8),
        vector_source=vector_source,
        nl_parser=parser,
    )
    source.execute_query_nl("find eligibility info", collection="contracts")
    assert vector_source.knn.call_args.kwargs["index_name"] == "contracts"
