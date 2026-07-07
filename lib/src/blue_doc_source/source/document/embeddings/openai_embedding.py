from typing import Dict, List

from openai import OpenAI

from .base import EmbeddingModel

_MODEL_DIMS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingModel(EmbeddingModel):
    """Generates embeddings via the OpenAI Embeddings API.

    Uses ``text-embedding-3-small`` by default (1536-dim). Pass a different
    ``model`` to switch, but ensure the target Redis index was created with
    the matching dimension.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
                environment variable when omitted.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    @property
    def dim(self) -> int:
        if self.model not in _MODEL_DIMS:
            raise ValueError(
                f"Unknown embedding model {self.model!r}. "
                f"Known models: {list(_MODEL_DIMS)}"
            )
        return _MODEL_DIMS[self.model]

    def embed_text(self, text: str) -> List[float]:  # pragma: no cover
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
