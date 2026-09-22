from collections.abc import Mapping
from typing import Any, ClassVar, Self

from giskard.agents import BaseEmbeddingModel, BaseGenerator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..settings import (
    get_default_embedding_model,
    get_default_generator,
    get_default_judge,
)
from .judge import BaseJudge, OptionalJudgeInput


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
    serialization so persisted checks store ``judge`` only. Assignment and
    ``model_copy(update={"generator": ...})`` also remigrate so ``_judge``
    never silently falls back to the default while a live generator is set.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    judge: OptionalJudgeInput = Field(
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

    @model_validator(mode="after")
    def _consume_legacy_generator_field(self) -> Self:
        # validate_assignment re-applies the assigned ``generator`` after the
        # before-validator remaps it onto ``judge``, and may leave ``judge`` as
        # a raw ``BaseGenerator`` without running OptionalJudgeInput coercion.
        if self.generator is None:
            return self
        if isinstance(self.judge, BaseJudge):
            object.__setattr__(self, "generator", None)
            return self
        raw = self.judge if self.judge is not None else self.generator
        object.__setattr__(self, "judge", BaseJudge.parse(raw))
        object.__setattr__(self, "generator", None)
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        # model_copy does not re-run validators; remigrate generator→judge here.
        patch: dict[str, Any] | None = dict(update) if update is not None else None
        if patch is not None and patch.get("generator") is not None:
            resulting_judge = patch["judge"] if "judge" in patch else self.judge
            if resulting_judge is not None:
                raise ValueError("Cannot provide both 'generator' and 'judge'")
            patch["judge"] = BaseJudge.parse(patch.pop("generator"))
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
