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


def test_execute_query_nl_dispatches_ingest_with_opt_in(tmp_path):
    # Write ops require explicit allow_write_ops=True (#79).
    path = make_txt(tmp_path)
    parser = NLQueryParser(parse_fn=lambda text: {"op": "DOCUMENT.INGEST", "path": path})
    source = DocumentDataSource(chunk_size=5, overlap=1, nl_parser=parser)
    records = source.execute_query_nl(f"ingest {path}", allow_write_ops=True)
    assert len(records) > 0


def test_execute_query_nl_rejects_write_ops_by_default(tmp_path):
    # Regression (#79): untrusted NL input must not trigger state-changing ops.
    path = make_txt(tmp_path)
    for op_dict in (
        {"op": "DOCUMENT.INGEST", "path": path},
        {"op": "DOCUMENT.DELETE", "path": path},
        {"op": "DOCUMENT.ENSURE_INDEX"},
    ):
        parser = NLQueryParser(parse_fn=lambda text, d=op_dict: d)
        source = DocumentDataSource(chunk_size=5, overlap=1, nl_parser=parser)
        with pytest.raises(ValueError, match="read-only"):
            source.execute_query_nl("please do it")


def test_execute_query_nl_rejects_invalid_op_dict():
    parser = NLQueryParser(parse_fn=lambda text: {"not_an_op": True})
    source = DocumentDataSource(nl_parser=parser)
    with pytest.raises(ValueError, match="invalid op dict"):
        source.execute_query_nl("whatever")


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


def test_execute_query_nl_dispatches_find(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    path = make_txt(tmp_path, "invoice number INV-001 vendor Acme total due $500")
    parser = NLQueryParser(
        parse_fn=lambda text: {
            "op": "DOCUMENT.FIND",
            "filter": {"document_type": "invoice"},
        }
    )
    source = DocumentDataSource(
        chunk_size=20,
        overlap=2,
        classifier=KeywordClassifier(),
        nl_parser=parser,
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    results = source.execute_query_nl("list all invoices")
    assert isinstance(results, list)


def test_execute_query_nl_dispatches_find_then_search(tmp_path):
    from blue_data_doc_source import KeywordClassifier
    content = (
        "invoice number due date amount due bill to line item payment "
        "subtotal tax total remit vendor quantity unit price "
    ) * 4
    path = make_txt(tmp_path, content)
    parser = NLQueryParser(
        parse_fn=lambda text: {
            "op": "DOCUMENT.FIND_THEN_SEARCH",
            "find_filter": {"document_type": "invoice"},
            "query": "payment due",
            "k": 3,
            "mode": "bm25",
        }
    )
    source = DocumentDataSource(
        chunk_size=20,
        overlap=2,
        classifier=KeywordClassifier(),
        nl_parser=parser,
    )
    source.execute_query({"op": "DOCUMENT.INGEST", "path": path})
    results = source.execute_query_nl("find payment terms in invoices")
    assert isinstance(results, list)
