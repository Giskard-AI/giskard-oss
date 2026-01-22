import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel, Field


def _parse_environ_or_default(env_var: str, default: int) -> int:
    try:
        return int(os.environ.get(env_var, default))
    except ValueError:
        return default


DEFAULT_MAX_BATCH_SIZE = _parse_environ_or_default(
    "GISKARD_AGENTS_DEFAULT_MAX_BATCH_SIZE", 1024
)
DEFAULT_MAX_TOTAL_CHARS = _parse_environ_or_default(
    "GISKARD_AGENTS_DEFAULT_MAX_TOTAL_CHARS", 20000
)


class EmbeddingParams(BaseModel):
    """Parameters for embedding model."""

    dimensions: int = Field(default=1536)


@discriminated_base
class BaseEmbeddingModel(Discriminated, ABC):
    params: EmbeddingParams = Field(default_factory=EmbeddingParams)

    @abstractmethod
    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]: ...

    async def embed(
        self,
        texts: list[str],
        params: EmbeddingParams | None = None,
        max_batch_size: int | None = None,
        max_total_chars: int | None = None,
    ) -> list[np.ndarray]:
        embedding_batches = []
        for batch in self.batched_embeddings(texts, max_batch_size, max_total_chars):
            embedding_batches.extend(await self._embed(batch, params))
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
            max_batch_size = DEFAULT_MAX_BATCH_SIZE
        if max_total_chars is None:
            max_total_chars = DEFAULT_MAX_TOTAL_CHARS

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
