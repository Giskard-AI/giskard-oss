from typing import Any

from giskard.agents import BaseEmbeddingModel, BaseGenerator
from pydantic import BaseModel, Field, model_validator

from ..settings import (
    get_default_embedding_model,
    get_default_generator,
    get_default_judge,
)
from .judge import BaseJudge


class WithGeneratorMixin(BaseModel):
    generator: BaseGenerator | None = Field(
        default=None,
        description="Generator for LLM evaluation. Defaults to the global default generator if None.",
    )

    @property
    def _generator(self) -> BaseGenerator:
        """Get the generator. If not set, return the global default generator."""
        return self.generator if self.generator is not None else get_default_generator()


class WithJudgeMixin(BaseModel):
    """Attach a :class:`~giskard.checks.core.judge.BaseJudge` to a check.

    ``generator`` remains accepted as a legacy constructor/dump alias and is
    rewritten to ``judge`` before field validation. It is excluded from
    serialization so persisted checks store ``judge`` only.
    """

    judge: BaseJudge | None = Field(
        default=None,
        description=(
            "Judge backend for evaluation. Defaults to the global default judge "
            "when None."
        ),
    )
    generator: BaseGenerator | None = Field(
        default=None,
        exclude=True,
        description=(
            "Legacy alias for an LLM generator judge. Prefer ``judge=``. "
            "Migrated to ``judge`` on construction and omitted from dumps."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_generator_to_judge(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "generator" not in data or data["generator"] is None:
            return data
        if "judge" in data and data["judge"] is not None:
            raise ValueError("Cannot provide both 'generator' and 'judge'")
        migrated = {key: value for key, value in data.items() if key != "generator"}
        migrated["judge"] = data["generator"]
        return migrated

    @property
    def _judge(self) -> BaseJudge:
        """Return the configured judge, or the global default."""
        return self.judge if self.judge is not None else get_default_judge()


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
