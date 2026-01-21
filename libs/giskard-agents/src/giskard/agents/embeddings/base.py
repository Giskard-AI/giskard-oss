from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel, Field


class EmbeddingParams(BaseModel):
    """Parameters for embedding model."""

    dimensions: int = Field(default=1536)
    params: dict[str, Any] = Field(
        default_factory=dict
    )  # Optional parameters for the embedding model (e.g. api_endpoint, api_key, etc.). Prefer using environment variables if available.
    max_batch_size: int = Field(default=1024)
    max_total_chars: int = Field(default=20_000)


@discriminated_base
class BaseEmbeddingModel(Discriminated, ABC):
    params: EmbeddingParams = Field(default_factory=EmbeddingParams)

    @abstractmethod
    async def _embed(self, texts: list[str]) -> list[np.ndarray]: ...

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        embedding_batches = []
        for batch in self.batched_embeddings(
            texts, self.params.max_batch_size, self.params.max_total_chars
        ):
            embedding_batches.extend(await self._embed(batch))
        return embedding_batches

    def batched_embeddings(
        self,
        texts: list[str],
        max_batch_size: int | None = None,
        max_total_chars: int | None = None,
    ) -> Iterator[list[str]]:
        """Batches texts for embedding process.

        This is modeled after the OpenAI API which sets limits both on batch size
        and total number of input tokens.
        """
        if max_batch_size is None:
            max_batch_size = self.params.max_batch_size
        if max_total_chars is None:
            max_total_chars = self.params.max_total_chars

        current_batch = []

        for text in texts:
            # If adding text item would exceed limits, yield current batch
            if (len(current_batch) >= max_batch_size) or (
                sum(len(t) for t in current_batch) + len(text) > max_total_chars
            ):
                if current_batch:
                    yield current_batch
                # Prevent a single too long document to make embeddings fail
                current_batch = [text[:max_total_chars]]
            else:
                current_batch.append(text)

        # Yield remaining items if present
        if current_batch:
            yield current_batch
