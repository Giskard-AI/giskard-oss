from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

from giskard.agents import BaseEmbeddingModel, BaseGenerator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..settings import (
    get_default_embedding_model,
    get_default_generator,
    get_default_judge,
)
from .judge import BaseJudge, LLMChatJudge, OptionalJudgeInput


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

    ``generator`` remains accepted as a legacy constructor alias and mutable
    compatibility property. It is backed by ``judge`` so there is only one
    stored source of truth.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    judge: OptionalJudgeInput = Field(
        default=None,
        description=(
            "Judge backend for evaluation. Defaults to the global default judge "
            "when None."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_generator_to_judge(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "generator" not in data:
            return data
        if "judge" in data:
            raise ValueError("Cannot provide both 'generator' and 'judge'")
        migrated = {key: value for key, value in data.items() if key != "generator"}
        if data["generator"] is not None:
            migrated["judge"] = data["generator"]
        return migrated

    if TYPE_CHECKING:
        # Keep the legacy constructor/assignment API visible to static tooling
        # without creating a second Pydantic field at runtime.
        generator: BaseGenerator | None = None
    else:

        @property
        def generator(self) -> BaseGenerator | None:
            """Legacy generator view backed by the configured LLM judge."""
            if isinstance(self.judge, LLMChatJudge):
                return self.judge.generator
            return None

        @generator.setter
        def generator(self, value: BaseGenerator | None) -> None:
            """Replace or clear the configured judge through the legacy API."""
            self.judge = None if value is None else BaseJudge.parse(value)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        # model_copy does not re-run validators; remigrate generator→judge here.
        patch: dict[str, Any] | None = dict(update) if update is not None else None
        if patch is not None:
            if "generator" in patch and "judge" in patch:
                raise ValueError("Cannot provide both 'generator' and 'judge'")
            if "generator" in patch:
                patch["judge"] = patch.pop("generator")
            if patch.get("judge") is not None:
                patch["judge"] = BaseJudge.parse(patch["judge"])
        return super().model_copy(update=patch, deep=deep)

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
