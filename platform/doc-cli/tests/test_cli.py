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
        result = runner.invoke(cli, ["ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        assert "test" in result.output

    def test_json_output(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "json", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ready"
        assert data["collection"] == "test"

    def test_nl_output(self, runner, pipeline_env):
        result = runner.invoke(cli, ["--format", "nl", "ensure-index", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ready"

    def test_idempotent(self, runner, pipeline_env):
        runner.invoke(cli, ["ensure-index", "--collection", "test"])
        result = runner.invoke(cli, ["ensure-index", "--collection", "test"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_txt(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        assert "document_id" in result.output
        assert "chunks" in result.output

    def test_ingest_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "json", "ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "document_id" in data
        assert data["chunk_count"] >= 1

    def test_ingest_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "nl", "ingest", txt_file, "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "document_id" in data

    def test_ingest_missing_file(self, runner, pipeline_env):
        result = runner.invoke(cli, ["ingest", "/nonexistent/file.txt"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_ingest_returns_document_id_for_delete(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["--format", "json", "ingest", txt_file, "--collection", "test"])
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
        result = runner.invoke(cli, ["search", "introduction", "--collection", "test", "-k", "2"])
        assert result.exit_code == 0

    def test_search_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["--format", "json", "search", "introduction", "--collection", "test"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_search_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["--format", "nl", "search", "introduction", "--collection", "test", "-k", "2"])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        for line in lines:
            json.loads(line)  # each line must be valid JSON

    def test_search_no_results(self, runner, pipeline_env):
        pipeline_env.ensure_index(collection="test")
        result = runner.invoke(cli, ["search", "anything", "--collection", "test"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_table_output_has_header(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        pipeline_env.ingest(txt_file, collection="test")
        result = runner.invoke(cli, ["search", "introduction", "--collection", "test"])
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
            cli, ["--format", "json", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_missing_args(self, runner, pipeline_env):
        result = runner.invoke(cli, ["delete"])
        assert result.exit_code != 0
        assert "--path" in result.output or "error" in result.output.lower()

    def test_delete_id_without_chunks(self, runner, pipeline_env):
        result = runner.invoke(cli, ["delete", "--id", "doc_abc123"])
        assert result.exit_code != 0

    def test_delete_json_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "--format", "json", "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        out = json.loads(result.output)
        assert "deleted" in out
        assert out["deleted"] >= 0

    def test_delete_nl_output(self, runner, pipeline_env, txt_file):
        pipeline_env.ensure_index(collection="test")
        ingest_result = runner.invoke(
            cli, ["--format", "json", "ingest", txt_file, "--collection", "test"]
        )
        data = json.loads(ingest_result.output)

        result = runner.invoke(cli, [
            "--format", "nl", "delete",
            "--id", data["document_id"],
            "--chunks", str(data["chunk_count"]),
            "--collection", "test",
        ])
        assert result.exit_code == 0
        out = json.loads(result.output)
        assert "deleted" in out
