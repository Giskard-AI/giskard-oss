from .base import BaseEmbeddingModel
from .litellm_embedding_model import LitellmEmbeddingModel

# Default embedding model uses Litellm
EmbeddingModel = LitellmEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "LitellmEmbeddingModel",
    "EmbeddingModel",
]
