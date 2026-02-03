import numpy as np
from litellm import aembedding
from pydantic import Field

from .base import BaseEmbeddingModel, EmbeddingParams


@BaseEmbeddingModel.register("litellm")
class LitellmEmbeddingModel(BaseEmbeddingModel):
    """An embedding model that uses Litellm."""

    model: str = Field(default="gemini/gemini-embedding-001")

    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        params_ = self.params.model_dump()

        if params is not None:
            params_.update(params.model_dump(exclude_unset=True))

        result = await aembedding(
            model=self.model,
            input=texts,
            **params_,
        )
        embeddings = [np.array(elt["embedding"]) for elt in result.data]
        return embeddings
