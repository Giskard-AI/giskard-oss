import importlib
from typing import override

import numpy as np
from pydantic import Field

from .base import BaseEmbeddingModel, EmbeddingParams


@BaseEmbeddingModel.register("sentence-transformers")
class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """An embedding model backed by the sentence-transformers library.

    Uses local embedding models from the `sentence-transformers` package, enabling
    offline embedding generation without any API keys. This is useful for CI/CD
    pipelines, cost-sensitive deployments, and offline evaluation.

    The ``sentence-transformers`` package is an optional dependency. Install it with:

    ```bash
    pip install giskard-agents[local-embeddings]
    # or directly:
    pip install sentence-transformers
    ```

    Parameters
    ----------
    model : str
        The sentence-transformers model name or path. Defaults to
        ``"all-MiniLM-L6-v2"``, a fast, lightweight model (384 dimensions) that
        works well on CPU.

        Other recommended options:
        - ``"all-mpnet-base-v2"`` — higher quality, 768 dimensions
        - ``"multi-qa-MiniLM-L6-cos-v1"`` — optimized for semantic search

    Examples
    --------
    >>> from giskard.agents.embeddings import SentenceTransformerEmbedding
    >>> from giskard.checks import set_default_embedding_model
    >>>
    >>> # Use local embeddings (no API key needed)
    >>> model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
    >>> set_default_embedding_model(model)
    """

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="The sentence-transformers model name or path.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Validate that sentence-transformers is available
        self._get_st_model()

    @staticmethod
    def _get_st_model():
        """Lazily import and return the SentenceTransformer class.

        Raises
        ------
        ImportError
            If sentence-transformers is not installed, with a clear
            installation hint.
        """
        try:
            from sentence_transformers import SentenceTransformer as ST
        except ImportError:
            raise ImportError(
                "The `sentence-transformers` package is required for "
                "SentenceTransformerEmbedding. Install it with:\n"
                "  pip install sentence-transformers\n"
                "Or install giskard-agents with the local-embeddings extra:\n"
                "  pip install giskard-agents[local-embeddings]"
            ) from None
        return ST

    @override
    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        """Generate embeddings for the given texts using a local model.

        Parameters
        ----------
        texts : list[str]
            List of text strings to embed.
        params : EmbeddingParams | None
            Optional embedding parameters (dimensions are inferred from
            the model output).

        Returns
        -------
        list[np.ndarray]
            List of embedding vectors, one per input text.
        """
        ST = self._get_st_model()
        model = ST(self.model)
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Ensure we return a list of numpy arrays
        if isinstance(embeddings, np.ndarray):
            return [embeddings[i] for i in range(embeddings.shape[0])]
        return [np.array(e) for e in embeddings]
