import pytest
from unittest.mock import MagicMock, patch

from blue_doc_source.source.document.embeddings.openai_embedding import OpenAIEmbeddingModel


@patch("blue_doc_source.source.document.embeddings.openai_embedding.OpenAI")
def test_dim_small(mock_openai):
    model = OpenAIEmbeddingModel("text-embedding-3-small")
    assert model.dim == 1536


@patch("blue_doc_source.source.document.embeddings.openai_embedding.OpenAI")
def test_dim_large(mock_openai):
    model = OpenAIEmbeddingModel("text-embedding-3-large")
    assert model.dim == 3072


@patch("blue_doc_source.source.document.embeddings.openai_embedding.OpenAI")
def test_dim_ada(mock_openai):
    model = OpenAIEmbeddingModel("text-embedding-ada-002")
    assert model.dim == 1536


@patch("blue_doc_source.source.document.embeddings.openai_embedding.OpenAI")
def test_dim_unknown_model_raises(mock_openai):
    model = OpenAIEmbeddingModel("gpt-4o")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        _ = model.dim
