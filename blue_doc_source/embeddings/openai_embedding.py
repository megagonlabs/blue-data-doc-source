from typing import List

from openai import OpenAI

from .base import EmbeddingModel


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

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
