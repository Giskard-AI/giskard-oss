import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


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
DEFAULT_MAX_CONCURRENT_EMBEDDING_BATCHES = _parse_environ_or_default(
    "GISKARD_AGENTS_DEFAULT_MAX_CONCURRENT_EMBEDDING_BATCHES", 4
)


class EmbeddingParams(BaseModel):
    """Parameters for embedding model."""

    dimensions: int = Field(default=1536)


@discriminated_base
class BaseEmbeddingModel(Discriminated, ABC):
    """Base class for embedding models.

    Unknown fields are rejected (``extra="forbid"``). Without this, pydantic
    silently drops unrecognized keys: because concrete models default their
    ``model`` field, a persisted check carrying a typo'd key (``model_name``
    instead of ``model``) would embed with the default model and every
    similarity score would be computed against the wrong vector space.
    """

    # Rationale and the subclass rule: see ``Discriminated`` in giskard-core.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

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
        max_concurrency: int | None = None,
    ) -> list[np.ndarray]:
        """Embed texts, sending the batches concurrently.

        Parameters
        ----------
        texts : list[str]
            Texts to embed, in order.
        params : EmbeddingParams, optional
            Embedding parameters passed to the underlying model.
        max_batch_size : int, optional
            Maximum number of texts per embedding request.
        max_total_chars : int, optional
            Maximum total number of characters per embedding request.
            Texts longer than this are truncated (see
            :meth:`batched_embeddings`).
        max_concurrency : int, optional
            Maximum number of embedding requests in flight at once.
            Defaults to ``DEFAULT_MAX_CONCURRENT_EMBEDDING_BATCHES``.

        Returns
        -------
        list[np.ndarray]
            One embedding per input text, in input order.
        """
        if max_concurrency is None:
            max_concurrency = DEFAULT_MAX_CONCURRENT_EMBEDDING_BATCHES
        batches = list(self.batched_embeddings(texts, max_batch_size, max_total_chars))
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def embed_batch(batch: list[str]) -> list[np.ndarray]:
            async with semaphore:
                return await self._embed(batch, params)

        # asyncio.gather preserves order, so the flattened result lines up
        # with the input texts even though batches run concurrently.
        results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        return [embedding for batch in results for embedding in batch]

    def batched_embeddings(
        self,
        texts: list[str],
        max_batch_size: int | None = None,
        max_total_chars: int | None = None,
    ) -> Iterator[list[str]]:
        """Batches texts for embedding process.

        This is modeled after the OpenAI API which sets limits both on batch size
        and total number of input tokens.

        A text longer than ``max_total_chars`` is truncated to its first
        ``max_total_chars`` characters so the embedding request does not fail;
        a warning is logged once the batches are exhausted, reporting how many
        texts were truncated.
        """
        if max_batch_size is None:
            max_batch_size = DEFAULT_MAX_BATCH_SIZE
        if max_total_chars is None:
            max_total_chars = DEFAULT_MAX_TOTAL_CHARS

        current_batch = []
        truncated_count = 0

        for text in texts:
            # If adding text item would exceed limits, yield current batch
            if (len(current_batch) >= max_batch_size) or (
                sum(len(t) for t in current_batch) + len(text) > max_total_chars
            ):
                if current_batch:
                    yield current_batch
                # Prevent a single too long document to make embeddings fail
                if len(text) > max_total_chars:
                    truncated_count += 1
                current_batch = [text[:max_total_chars]]
            else:
                current_batch.append(text)

        # Yield remaining items if present
        if current_batch:
            yield current_batch

        if truncated_count:
            logger.warning(
                "Truncated %d text(s) to %d characters each while batching "
                "embeddings; content beyond the limit was not embedded.",
                truncated_count,
                max_total_chars,
            )
