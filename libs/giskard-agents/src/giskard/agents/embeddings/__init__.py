from .base import BaseEmbeddingModel
from .litellm_embedding_model import LitellmEmbeddingModel
from .sentence_transformers_embedding import SentenceTransformerEmbedding

# Default embedding model uses Litellm
EmbeddingModel = LitellmEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "LitellmEmbeddingModel",
    "SentenceTransformerEmbedding",
    "EmbeddingModel",
]
