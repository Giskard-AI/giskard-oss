"""Factory helpers for picking an embedding backend from a model string."""

from giskard.llm import supports_native

from .base import BaseEmbeddingModel
from .litellm_embedding_model import LitellmEmbeddingModel


def resolve_embedding_model(model: str) -> BaseEmbeddingModel:
    """Return the best embedding backend for *model*.

    Uses ``LitellmEmbeddingModel`` (giskard-llm) when the provider is supported
    natively; otherwise falls back to ``LiteLLMEmbeddingModel`` (litellm package).

    Parameters
    ----------
    model
        Embedding model identifier (e.g. ``text-embedding-3-small`` or
        ``deepseek/deepseek-embed``).

    Returns
    -------
    BaseEmbeddingModel
        An embedding model configured for *model*.

    Raises
    ------
    ImportError
        When the native path is unavailable and the optional ``litellm`` extra
        is not installed.
    """
    if supports_native(model, "embedding"):
        return LitellmEmbeddingModel(model=model)

    from .litellm_package_embedding_model import LiteLLMEmbeddingModel

    return LiteLLMEmbeddingModel(model=model)
