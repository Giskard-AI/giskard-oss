from giskard.agents import BaseEmbeddingModel, BaseGenerator
from pydantic import BaseModel, Field

from ..settings import get_default_embedding_model, get_default_generator
from .suite_usage import get_active_suite_usage_collector, with_usage_recording


class WithGeneratorMixin(BaseModel):
    generator: BaseGenerator | None = Field(
        default=None,
        description="Generator for LLM evaluation. Defaults to the global default generator if None.",
    )

    @property
    def _generator(self) -> BaseGenerator:
        """Get the generator. If not set, return the global default generator."""
        generator = (
            self.generator if self.generator is not None else get_default_generator()
        )
        if get_active_suite_usage_collector() is None:
            return generator
        return with_usage_recording(generator)


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
