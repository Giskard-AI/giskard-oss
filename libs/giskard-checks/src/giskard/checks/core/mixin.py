from collections.abc import Mapping
from typing import Any, ClassVar, Self

from giskard.agents import BaseEmbeddingModel, BaseGenerator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticUndefined

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


class WithJudgeMixin(WithGeneratorMixin):
    """Attach a :class:`~giskard.checks.core.judge.BaseJudge` to a check.

    ``generator`` remains accepted as a legacy constructor alias and mutable
    compatibility property. It is backed by ``judge`` so there is only one
    stored source of truth.

    Preserve ``WithGeneratorMixin`` inheritance and its ``_generator`` accessor
    for existing consumers that identify and inject generators through them.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    judge: OptionalJudgeInput = Field(
        default=None,
        description=(
            "Judge backend for evaluation. Defaults to the global default judge "
            "when None."
        ),
    )
    generator: BaseGenerator | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_generator_to_judge(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        if info.field_name is not None:
            return data
        if "generator" in data and "judge" in data:
            raise ValueError("Cannot provide both 'generator' and 'judge'")

        migrated = dict(data)
        if "judge" in migrated:
            judge = migrated["judge"]
            if judge is not None:
                judge = BaseJudge.parse(judge)
                migrated["judge"] = judge
            # Serialized LLM judges must satisfy non-nullable legacy fields too.
            migrated["generator"] = (
                judge.generator if isinstance(judge, LLMChatJudge) else None
            )
            return migrated

        if "generator" in migrated:
            generator = migrated["generator"]
        else:
            generator_field = cls.model_fields["generator"]
            if generator_field.default_factory_takes_validated_data:
                raise TypeError(
                    "WithJudgeMixin cannot migrate a generator default_factory "
                    "that takes validated data; use a fixed default or a "
                    "zero-argument factory"
                )
            generator = generator_field.get_default(
                call_default_factory=True, validated_data=migrated
            )
            if generator is PydanticUndefined:
                return migrated
            migrated["generator"] = generator
            if generator is None:
                # An omitted default is not an explicit judge override.
                return migrated

        migrated["judge"] = generator
        return migrated

    @model_validator(mode="after")
    def _sync_generator_from_judge(self) -> Self:
        generator = (
            self.judge.generator if isinstance(self.judge, LLMChatJudge) else None
        )
        object.__setattr__(self, "generator", generator)
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_legacy_generator(self, handler: Any) -> dict[str, Any]:
        serialized = handler(self)
        serialized.pop("generator", None)
        return serialized

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "generator":
            name = "judge"
            value = None if value is None else BaseJudge.parse(value)
        super().__setattr__(name, value)

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
                generator = patch.pop("generator")
                # Legacy model_copy updates are trusted, just like BaseModel's.
                patch["judge"] = (
                    LLMChatJudge.model_construct(generator=generator)
                    if generator is not None
                    else None
                )
            elif patch.get("judge") is not None:
                patch["judge"] = BaseJudge.parse(patch["judge"])
        copied = super().model_copy(update=patch, deep=deep)
        generator = (
            copied.judge.generator if isinstance(copied.judge, LLMChatJudge) else None
        )
        object.__setattr__(copied, "generator", generator)
        return copied

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
