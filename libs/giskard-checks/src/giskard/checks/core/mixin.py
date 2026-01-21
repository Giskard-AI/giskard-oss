from giskard.agents.embeddings import BaseEmbeddingModel
from giskard.agents.generators.base import BaseGenerator
from pydantic import BaseModel, Field

from ..settings import get_default_embedding_model, get_default_generator


class WithGeneratorMixin(BaseModel):
    generator: BaseGenerator = Field(
        default_factory=get_default_generator,
        exclude=True,  # Not serializable
        description="Generator for LLM evaluation",
    )


class WithEmbeddingMixin(BaseModel):
    embedding_model: BaseEmbeddingModel = Field(
        default_factory=get_default_embedding_model,
        exclude=True,  # Not serializable
        description="Embedding model for embedding text",
    )
