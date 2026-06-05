from typing import List

from openai import OpenAI

from .base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
