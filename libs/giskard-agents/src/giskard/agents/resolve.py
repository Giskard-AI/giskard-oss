"""Resolve default generator and embedding backends from model strings."""

from giskard.llm import supports_native

from .embeddings.base import BaseEmbeddingModel
from .embeddings.giskard_llm_embedding_model import GiskardLLMEmbeddingModel
from .embeddings.litellm_embedding_model import LiteLLMEmbeddingModel
from .generators.base import BaseGenerator
from .generators.giskard_llm_generator import GiskardLLMGenerator


def resolve_generator(model: str) -> BaseGenerator:
    """Return the best generator backend for *model*."""
    if supports_native(model, "completion"):
        return GiskardLLMGenerator(model=model)
    from .generators.litellm_generator import LiteLLMGenerator

    return LiteLLMGenerator(model=model)


def resolve_embedding_model(model: str) -> BaseEmbeddingModel:
    """Return the best embedding backend for *model*."""
    if supports_native(model, "embedding"):
        return GiskardLLMEmbeddingModel(model=model)
    return LiteLLMEmbeddingModel(model=model)
