from .base import BaseEmbeddingModel
from .litellm_embedding_model import LitellmEmbeddingModel
from .litellm_package_embedding_model import LiteLLMEmbeddingModel

# Default embedding model uses giskard-llm (via LitellmEmbeddingModel)
EmbeddingModel = LitellmEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "LitellmEmbeddingModel",
    "LiteLLMEmbeddingModel",
    "EmbeddingModel",
]
