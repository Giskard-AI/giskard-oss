import asyncio
from typing import Any

import numpy as np
from giskard.agents.embeddings import BaseEmbeddingModel
from giskard.agents.embeddings.base import EmbeddingParams
from pydantic import Field, PrivateAttr


@BaseEmbeddingModel.register("sentence_transformer")
class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local embedding model backed by `sentence-transformers`.

    This provider enables offline or API-key-free semantic similarity checks.
    Install it with the optional dependency:

    ```bash
    pip install "giskard-checks[local-embeddings]"
    ```
    """

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="The sentence-transformers model name to load locally.",
    )
    device: str | None = Field(
        default=None,
        description="Optional device override passed to SentenceTransformer.",
    )
    normalize_embeddings: bool = Field(
        default=False,
        description="Whether to normalize embeddings before cosine similarity.",
    )

    _model: Any = PrivateAttr(default=None)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedding requires the optional dependency "
                "'sentence-transformers'. Install it with "
                "'pip install \"giskard-checks[local-embeddings]\"'."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device

        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [np.asarray(embedding) for embedding in embeddings]

    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        _ = params  # Sentence-transformers controls dimensionality via model choice.
        return await asyncio.to_thread(self._encode, texts)
