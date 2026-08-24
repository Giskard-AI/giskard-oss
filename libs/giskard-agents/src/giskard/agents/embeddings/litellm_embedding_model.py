"""Optional LiteLLM-package embedding model.

Install the optional dependency with:

    pip install giskard-agents[litellm]

Importing this module is safe without litellm; instantiation raises ImportError.
"""

from typing import Any

import numpy as np
from pydantic import Field

from .base import BaseEmbeddingModel, EmbeddingParams


def _import_litellm() -> Any:
    try:
        import litellm

        return litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLMEmbeddingModel requires the optional 'litellm' dependency. "
            "Install it with: pip install giskard-agents[litellm]"
        ) from exc


@BaseEmbeddingModel.register("litellm_package")
class LiteLLMEmbeddingModel(BaseEmbeddingModel):
    """An embedding model backed by the LiteLLM Python package."""

    model: str = Field(default="text-embedding-3-small")

    def model_post_init(self, __context: Any) -> None:
        """Fail fast if litellm is not installed."""
        super().model_post_init(__context)
        _import_litellm()

    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        litellm = _import_litellm()
        params_ = self.params.model_dump() if self.params is not None else {}

        if params is not None:
            params_.update(params.model_dump(exclude_unset=True))

        result = await litellm.aembedding(
            model=self.model,
            input=texts,
            **params_,
        )
        data = result.data if hasattr(result, "data") else result["data"]
        return [
            np.array(elt.embedding if hasattr(elt, "embedding") else elt["embedding"])
            for elt in data
        ]
