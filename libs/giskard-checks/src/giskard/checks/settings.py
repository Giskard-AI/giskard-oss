"""Runtime and environment configuration for giskard-checks."""

from giskard.agents import BaseEmbeddingModel, BaseGenerator, EmbeddingModel, Generator
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Runtime overrides (take precedence over environment settings)
_default_generator: BaseGenerator | None = None
_default_embedding_model: BaseEmbeddingModel | None = None

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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
        description="Default LLM model identifier for checks without an explicit generator.",
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
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        return None


def get_settings() -> GiskardChecksSettings:
    """Return settings loaded from the environment."""
    return GiskardChecksSettings()


def set_default_generator(generator: BaseGenerator) -> None:
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator
        The generator to use as default for all LLM checks.
    """
    global _default_generator
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


def set_default_embedding_model(embedding_model: BaseEmbeddingModel) -> None:
    """Set the default embedding model for all checks.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        The embedding model to use as default for all embedding checks.
    """
    global _default_embedding_model
    _default_embedding_model = embedding_model


def get_default_embedding_model() -> BaseEmbeddingModel:
    """Get the current default embedding model.

    Returns
    -------
    BaseEmbeddingModel
        The runtime override if set, otherwise a model built from
        :envvar:`GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL`, or text-embedding-3-small.
    """
    if _default_embedding_model is not None:
        return _default_embedding_model
    return EmbeddingModel(model=get_settings().default_embedding_model)
