import numpy as np
from litellm import aembedding
from pydantic import Field

from .base import BaseEmbeddingModel


@BaseEmbeddingModel.register("litellm")
class LitellmEmbeddingModel(BaseEmbeddingModel):
    """An embedding model that uses Litellm."""

    model: str = Field(default="gemini/gemini-embedding-001")

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        result = await aembedding(
            model=self.model,
            input=texts,
            dimensions=self.params.dimensions,
            **self.params.params,
        )
        embeddings = [np.array(elt["embedding"]) for elt in result.data]
        return embeddings
