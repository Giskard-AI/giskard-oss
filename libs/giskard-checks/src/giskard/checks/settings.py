"""Runtime and environment configuration for giskard-checks."""

from giskard.agents import (
    BaseEmbeddingModel,
    BaseGenerator,
    BaseSOM,
    EmbeddingModel,
    Generator,
)
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.judge import BaseJudge, LLMChatJudge

_default_generator: BaseGenerator | None = None
_default_judge: BaseJudge | None = None
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
    default_judge: str | None = Field(
        default=None,
        description=(
            "Default judge identifier or JSON dump. Supports provider/model "
            "strings (with optional llm/ or som/ kind prefix) or a JSON object. "
            "When unset, judges fall back to the default LLM generator."
        ),
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


def set_default_judge(
    judge: BaseJudge | BaseGenerator | BaseSOM | str | None,
) -> None:
    """Set the default judge independently of content generation.

    Parameters
    ----------
    judge : BaseJudge, BaseGenerator, BaseSOM, str, or None
        A configured judge, generator, SOM model, or provider/model identifier
        (optionally prefixed with ``llm/`` or ``som/``). JSON object strings are
        also accepted. ``None`` clears the runtime override so the environment
        default or LLM generator fallback is used.

    Examples
    --------
    Keep content generation on an LLM and use Jev for evaluation::

        from giskard.checks import set_default_generator, set_default_judge
        set_default_generator("azure_ai/gpt-5.6-luna")
        set_default_judge("typesafe/jev")
    """
    global _default_judge
    if judge is None:
        _default_judge = None
        return
    _default_judge = BaseJudge.parse(judge)


def get_default_judge() -> BaseJudge:
    """Return the configured judge, falling back to the default LLM generator.

    Resolution order:

    1. Runtime override from :func:`set_default_judge`
    2. :envvar:`GISKARD_CHECKS_DEFAULT_JUDGE` when set
    3. An :class:`~giskard.checks.core.judge.LLMChatJudge` that tracks the
       default generator
    """
    if _default_judge is not None:
        return _default_judge
    configured = get_settings().default_judge
    if configured is not None and configured.strip():
        return BaseJudge.parse(configured)
    return LLMChatJudge()


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
