"""CLI tests — use MemoryVectorSource + DummyEmbeddingModel; no external services."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from blue_data_doc_cli.config import reset_pipeline
from blue_data_doc_cli.main import cli


@pytest.fixture(autouse=True)
def _reset():
    reset_pipeline()
    yield
    reset_pipeline()


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def pipeline_env(monkeypatch):
    """Patch get_pipeline to return a real pipeline with in-memory vector store."""
    from blue_data_doc_source import BasicEnricher, DocumentPipeline, DummyEmbeddingModel
    from blue_data.source.vector.memory_vector_source import MemoryVectorSource

    vector_source = MemoryVectorSource()
    pipeline = DocumentPipeline(
        chunk_size=200,
        overlap=20,
        embedding_model=DummyEmbeddingModel(),
        vector_source=vector_source,
        enricher=BasicEnricher(),
        collection="test",
    )
    with patch("blue_data_doc_cli.main.get_pipeline", return_value=pipeline):
        yield pipeline


@pytest.fixture()
def txt_file(tmp_path):
    """Temporary .txt file with enough text to produce at least one chunk."""
    content = (
        "Introduction\n\n"
        + ("This is a test document with enough text to be chunked. " * 20)
        + "\n\nConclusion\n\n"
        + ("Final section with more text. " * 10)
    )
    p = tmp_path / "sample.txt"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# ensure-index
# ---------------------------------------------------------------------------

class TestEnsureIndex:
    def test_creates_index(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        assert "test" in result.output

    def test_json_output(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "doc", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ready"
        assert data["collection"] == "test"

    def test_nl_output(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "nl", "doc", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ready"

    def test_idempotent(self, runner, pipeline_env):
        runner.invoke(cli, ["doc", "ensure-index", "--collection", "test"])
        result = runner.invoke(cli, ["doc", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_txt(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["doc", "ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        assert "document_id" in result.output
        assert "chunks" in result.output

    def test_ingest_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "document_id" in data
        assert data["chunk_count"] >= 1

    def test_ingest_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "nl", "doc", "ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "document_id" in data

    def test_ingest_missing_file(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "ingest", "/nonexistent/file.txt"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_ingest_json_error_is_structured(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "doc", "ingest", "/nonexistent/file.txt"])
        assert result.exit_code != 0
        err = json.loads(result.output)
        assert err["error"]["code"] == "NOT_FOUND"
        assert "message" in err["error"]

    def test_ingest_returns_document_id_for_delete(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"])
        data = json.loads(result.output)
        assert data["document_id"].startswith("doc_")
        assert data["chunk_count"] > 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_results(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["doc", "search", "introduction", "--collection", "test", "-k", "2"])
        assert result.exit_code == 0

    def test_search_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["--format", "json", "doc", "search", "introduction", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_search_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["--format", "nl", "doc", "search", "introduction", "--collection", "test", "-k", "2"])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        for line in lines:
            json.loads(line)  # each line must be valid JSON

    def test_search_no_results(self, runner, pipeline_env):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["doc", "search", "anything", "--collection", "test"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_table_output_has_header(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["doc", "search", "introduction", "--collection", "test"])
        assert result.exit_code == 0
        assert "file" in result.output
        assert "section" in result.output


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_by_id(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "doc", "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_missing_args(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "delete"])
        assert result.exit_code != 0
        assert "--path" in result.output or "error" in result.output.lower()

    def test_delete_id_without_chunks(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "delete", "--id", "doc_abc123"])
        assert result.exit_code != 0

    def test_delete_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "--format", "json", "doc", "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        out = json.loads(result.output)
        assert "deleted" in out
        assert out["deleted"] >= 0

    def test_delete_json_error_is_structured(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "doc", "delete"])
        assert result.exit_code != 0
        err = json.loads(result.output)
        assert err["error"]["code"] == "INVALID_ARGS"
        assert "message" in err["error"]

    def test_delete_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "--format", "nl", "doc", "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        out = json.loads(result.output)
        assert "deleted" in out


# ---------------------------------------------------------------------------
# search — new options
# ---------------------------------------------------------------------------

class TestSearchOptions:
    def test_search_mode_option_accepted(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["doc", "search", "introduction", "--collection", "test", "--mode", "vector"])
        assert result.exit_code == 0

    def test_search_context_chunks_option_accepted(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["doc", "search", "introduction", "--collection", "test", "--context-chunks", "2"])
        assert result.exit_code == 0

    def test_search_filter_option_accepted(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, [
            "doc", "search", "introduction", "--collection", "test",
            "--filter", '{"document_type": "unknown"}',
        ])
        assert result.exit_code == 0

    def test_search_invalid_filter_json_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "search", "query", "--filter", "not-json"])
        assert result.exit_code != 0

    def test_search_invalid_mode_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "search", "query", "--mode", "invalid"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_env_with_classifier(monkeypatch):
    """Pipeline with KeywordClassifier so DocumentRecords are written at ingest."""
    from blue_data_doc_source import BasicEnricher, DocumentPipeline, DummyEmbeddingModel, KeywordClassifier
    from blue_data.source.vector.memory_vector_source import MemoryVectorSource

    pipeline = DocumentPipeline(
        chunk_size=200,
        overlap=20,
        embedding_model=DummyEmbeddingModel(),
        vector_source=MemoryVectorSource(),
        enricher=BasicEnricher(),
        classifier=KeywordClassifier(),
        collection="test",
    )
    with patch("blue_data_doc_cli.main.get_pipeline", return_value=pipeline):
        yield pipeline


class TestFind:
    def test_find_returns_empty_without_records(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "find"])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_find_json_returns_empty_list_without_records(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "doc", "find"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_find_nl_returns_empty_without_records(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "nl", "doc", "find"])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_find_with_classifier_returns_records(self, runner, pipeline_env_with_classifier, txt_file):
        pipeline_env_with_classifier.ensure_index(collection="test")
        pipeline_env_with_classifier.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["--format", "json", "doc", "find"])
        assert result.exit_code == 0
        records = json.loads(result.output)
        assert isinstance(records, list)
        assert len(records) >= 1
        assert "document_id" in records[0]

    def test_find_invalid_filter_json_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "find", "--filter", "{bad json}"])
        assert result.exit_code != 0

    def test_find_filter_by_document_type(self, runner, pipeline_env_with_classifier, txt_file):
        pipeline_env_with_classifier.ensure_index(collection="test")
        pipeline_env_with_classifier.ingest(txt_file, collection="test")
        result = runner.invoke(cli, [
            "--format", "json", "doc", "find",
            "--filter", '{"document_type": "nonexistent_type"}',
        ])
        assert result.exit_code == 0
        assert json.loads(result.output) == []


# ---------------------------------------------------------------------------
# find-then-search
# ---------------------------------------------------------------------------

class TestFindThenSearch:
    def test_find_then_search_no_records_returns_empty(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, [
            "doc", "find-then-search", "introduction",
            "--filter", '{"document_type": "nonexistent"}',
            "--collection", "test",
        ])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_find_then_search_json_empty(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, [
            "--format", "json", "doc", "find-then-search", "introduction",
            "--filter", '{"document_type": "nonexistent"}',
            "--collection", "test",
        ])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_find_then_search_invalid_filter_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "find-then-search", "query", "--filter", "bad"])
        assert result.exit_code != 0

    def test_find_then_search_with_records_returns_chunks(self, runner, pipeline_env_with_classifier, txt_file):
        pipeline_env_with_classifier.ensure_index(collection="test")
        pipeline_env_with_classifier.ingest(txt_file, collection="test")
        result = runner.invoke(cli, [
            "--format", "json", "doc", "find-then-search", "introduction",
            "--collection", "test",
        ])
        assert result.exit_code == 0
        chunks = json.loads(result.output)
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# get-record
# ---------------------------------------------------------------------------

class TestGetRecord:
    def test_get_record_missing_document_id_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "get-record"])
        assert result.exit_code != 0

    def test_get_record_not_found_errors(self, runner, pipeline_env):
        result = runner.invoke(cli, ["doc", "get-record", "--document-id", "doc_nonexistent"])
        assert result.exit_code != 0
        assert "NOT_FOUND" in result.output or "not found" in result.output.lower()

    def test_get_record_json_not_found_is_structured(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "doc", "get-record", "--document-id", "doc_nonexistent"])
        assert result.exit_code != 0
        err = json.loads(result.output)
        assert err["error"]["code"] == "NOT_FOUND"

    def test_get_record_returns_record_after_ingest(self, runner, pipeline_env_with_classifier, txt_file):
        pipeline_env_with_classifier.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "doc", "ingest", txt_file, "--collection", "test"]
        )
        doc_id = json.loads(ingest_result.output)["document_id"]

        result = runner.invoke(cli, ["--format", "json", "doc", "get-record", "--document-id", doc_id])
        assert result.exit_code == 0
        record = json.loads(result.output)
        assert record["document_id"] == doc_id
