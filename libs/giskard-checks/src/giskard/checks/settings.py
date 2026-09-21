"""Runtime and environment configuration for giskard-checks."""

from typing import Literal

from giskard.agents import (
    BaseEmbeddingModel,
    BaseGenerator,
    BaseSOM,
    EmbeddingModel,
    Generator,
    resolve_som,
)
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._som import SOMJudgeGenerator

_default_generator: BaseGenerator | None = None
_default_judge: BaseGenerator | None = None
_default_embedding_model: BaseEmbeddingModel | None = None

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_REPORTED_FAILURES_ENV_VAR = "GISKARD_CHECKS_MAX_REPORTED_FAILURES"


class GiskardChecksSettings(BaseSettings):
    """Environment-backed settings for giskard-checks.

    Values can be set via environment variables prefixed with ``GISKARD_CHECKS_``
    or in a ``.env`` file at the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="GISKARD_CHECKS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_model: str = Field(
        default=DEFAULT_MODEL,
        description="Default model for content generation and judges without a separate default.",
    )
    default_embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Default embedding model identifier for checks without an explicit model.",
    )
    disable_rich_pretty: bool = Field(
        default=False,
        description="Disable rich.pretty installation for REPL output.",
    )
    max_reported_failures: int | None = Field(
        default=None,
        description="Maximum number of failures to include in suite reports. None means unlimited.",
    )

    @field_validator("max_reported_failures", mode="before")
    @classmethod
    def _normalize_max_reported_failures(cls, value: object) -> int | None:
        if value is None or value == "" or isinstance(value, bool):
            return None
        if not isinstance(value, (int, str)):
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None


def get_settings() -> GiskardChecksSettings:
    """Return settings loaded from the environment."""
    return GiskardChecksSettings()


def set_default_generator(generator: BaseGenerator | str) -> None:
    """Set the default model for content generation.

    Judges also use this generator until a separate default judge is configured.

    Parameters
    ----------
    generator : BaseGenerator or str
        The generator to use for content generation. A model
        identifier string (e.g. ``"openai/gpt-4o-mini"``) is wrapped in a
        :class:`~giskard.agents.Generator`.

    Examples
    --------
    Pass a model identifier::

        from giskard.checks import set_default_generator
        set_default_generator("openai/gpt-4o-mini")

    Pass a generator instance when you need extra configuration::

        from giskard.agents import Generator
        from giskard.checks import set_default_generator
        set_default_generator(Generator(model="openai/gpt-4o-mini"))
    """
    global _default_generator
    if isinstance(generator, str):
        _default_generator = Generator(model=generator)
    else:
        _default_generator = generator


def get_default_generator() -> BaseGenerator:
    """Get the current default generator.

    Returns
    -------
    BaseGenerator
        The runtime override if set, otherwise a generator built from
        :envvar:`GISKARD_CHECKS_DEFAULT_MODEL`, or a default GPT-4o-mini generator.
    """
    if _default_generator is not None:
        return _default_generator
    return Generator(model=get_settings().default_model)


def _create_judge_generator(
    model: str, model_type: Literal["llm", "som"] | None
) -> BaseGenerator:
    som = resolve_som(model)
    inferred_type = "som" if som is not None else "llm"
    if model_type not in (None, inferred_type):
        raise ValueError(
            "Use model_type='som' with a supported SOM provider, "
            "or 'llm' with an LLM provider"
        )
    return SOMJudgeGenerator(model=som) if som is not None else Generator(model=model)


def set_default_judge(
    judge: BaseGenerator | BaseSOM | str | None,
    *,
    model_type: Literal["llm", "som"] | None = None,
) -> None:
    """Set the default judge independently of content generation.

    Parameters
    ----------
    judge : BaseGenerator, BaseSOM, str, or None
        A configured generator, SOM model, or provider/model identifier.
        SOM models are adapted to pass/fail verdicts. ``None`` clears the
        runtime override, restoring the default generator.
    model_type : {"llm", "som"} or None, optional
        Model type for string identifiers. When omitted, providers supported
        by giskard.agents.som select a System One Model; other identifiers
        select an LLM. Explicit types must match the provider's capability.
        Configured models and ``None`` must omit this argument.

    Examples
    --------
    Keep content generation on an LLM and use Jev for evaluation::

        from giskard.checks import set_default_generator, set_default_judge
        set_default_generator("azure_ai/gpt-5.6-luna")
        set_default_judge("typesafe/jev")
    """
    global _default_judge
    if isinstance(judge, str):
        _default_judge = _create_judge_generator(judge, model_type)
        return
    if model_type is not None:
        raise ValueError("model_type is only supported with a model identifier string")
    if isinstance(judge, BaseSOM):
        judge = SOMJudgeGenerator(model=judge)
    if judge is not None and not isinstance(judge, BaseGenerator):
        raise TypeError(
            "judge must be a BaseGenerator, BaseSOM, model identifier string, or None"
        )
    _default_judge = judge


def get_default_judge() -> BaseGenerator:
    """Return the configured judge, falling back to the default generator."""
    return _default_judge if _default_judge is not None else get_default_generator()


def set_default_embedding_model(model: BaseEmbeddingModel | str | None) -> None:
    """Set the default embedding model independently of generation and judging.

    A model identifier is wrapped in :class:`~giskard.agents.EmbeddingModel`.
    Pass a configured :class:`~giskard.agents.BaseEmbeddingModel` for custom
    behavior, or ``None`` to restore the environment setting or built-in default.
    """
    global _default_embedding_model
    if isinstance(model, str):
        _default_embedding_model = EmbeddingModel(model=model)
    elif model is None or isinstance(model, BaseEmbeddingModel):
        _default_embedding_model = model
    else:
        raise TypeError(
            "model must be a BaseEmbeddingModel, model identifier string, or None"
        )


def get_default_embedding_model() -> BaseEmbeddingModel:
    """Get the current default embedding model.

    Returns
    -------
    BaseEmbeddingModel
        The runtime override if set, otherwise a model built from
        :envvar:`GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL`, or
        text-embedding-3-small by default.
    """
    if _default_embedding_model is not None:
        return _default_embedding_model
    return EmbeddingModel(model=get_settings().default_embedding_model)
