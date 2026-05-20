from giskard.agents.embeddings import BaseEmbeddingModel
from giskard.agents.generators.base import BaseGenerator
from pydantic import BaseModel, Field

from ..settings import get_default_embedding_model, get_default_generator


class WithGeneratorMixin(BaseModel):
    generator: BaseGenerator | None = Field(
        default=None,
        description="Generator for LLM evaluation. Defaults to the global default generator if None.",
    )

    @property
    def _generator(self) -> BaseGenerator:
        """Get the generator. If not set, return the global default generator."""
        return self.generator if self.generator is not None else get_default_generator()


class WithEmbeddingMixin(BaseModel):
    embedding_model: BaseEmbeddingModel | None = Field(
        default=None,
        description="Embedding model for embedding text. Defaults to the global default if None.",
    )

    @property
    def _embedding_model(self) -> BaseEmbeddingModel:
        """Get the embedding model. If not set, return the global default embedding model."""
        return (
            self.embedding_model
            if self.embedding_model is not None
            else get_default_embedding_model()
        )
