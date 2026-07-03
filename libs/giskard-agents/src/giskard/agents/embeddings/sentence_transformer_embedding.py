"""Optional SentenceTransformers-backed embedding model.

Install the optional dependency with::

    pip install giskard-agents[local-embeddings]

Importing this module is safe without sentence-transformers; instantiation raises
ImportError with an install hint if the optional dependency is missing.
"""

import asyncio
from typing import Any, override

import numpy as np
from pydantic import Field, PrivateAttr

from .base import BaseEmbeddingModel, EmbeddingParams


def _import_sentence_transformers() -> Any:
    try:
        import sentence_transformers

        return sentence_transformers
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformerEmbedding requires the optional "
            "'sentence-transformers' dependency. Install it with: "
            "pip install giskard-agents[local-embeddings]"
        ) from exc


@BaseEmbeddingModel.register("sentence_transformers")
class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Embedding model backed by the local ``sentence-transformers`` package."""

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model name or local path.",
    )
    device: str | None = Field(
        default=None,
        description="Optional device passed to SentenceTransformer, for example 'cpu' or 'cuda'.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to allow remote model code when loading from Hugging Face.",
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to SentenceTransformer.",
    )
    encode_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to SentenceTransformer.encode.",
    )

    _client: Any | None = PrivateAttr(default=None)

    def __init__(self, model: str | None = None, **data: Any) -> None:
        if model is not None:
            if "model" in data:
                raise TypeError(
                    "Pass the model name either positionally or by keyword, not both."
                )
            data["model"] = model
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        """Fail fast if sentence-transformers is not installed."""
        super().model_post_init(__context)
        _import_sentence_transformers()

    def _load_client(self) -> Any:
        if self._client is None:
            sentence_transformers = _import_sentence_transformers()
            self._client = sentence_transformers.SentenceTransformer(
                self.model,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                **self.model_kwargs,
            )
        return self._client

    @override
    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        client = self._load_client()
        encode_kwargs = {"convert_to_numpy": True, **self.encode_kwargs}

        embeddings = await asyncio.to_thread(
            client.encode,
            texts,
            **encode_kwargs,
        )
        return [np.asarray(embedding) for embedding in embeddings]
