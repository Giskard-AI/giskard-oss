from .base import BaseEmbeddingModel
from .giskard_llm_embedding_model import GiskardLLMEmbeddingModel
from .litellm_embedding_model import LiteLLMEmbeddingModel

EmbeddingModel = GiskardLLMEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "GiskardLLMEmbeddingModel",
    "LiteLLMEmbeddingModel",
    "EmbeddingModel",
]
