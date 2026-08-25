"""Runtime and environment configuration for giskard-checks."""

from giskard.agents import BaseGenerator, EmbeddingModel, Generator
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Runtime override (takes precedence over environment settings)
_default_generator: BaseGenerator | None = None

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
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator or str
        The generator to use as default for all LLM checks. A model
        identifier string (e.g. ``"openai/gpt-4o-mini"``) is wrapped in a
        :class:`~giskard.agents.generators.Generator`.

    Examples
    --------
    Pass a model identifier::

        from giskard.checks import set_default_generator
        set_default_generator("openai/gpt-4o-mini")

    Pass a generator instance when you need extra configuration::

        from giskard.agents.generators import Generator
        from giskard.checks import set_default_generator
        set_default_generator(Generator(model="openai/gpt-4o-mini"))
    """
    global _default_generator
    if isinstance(generator, str):
        from giskard.agents.generators import Generator

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


def get_default_embedding_model() -> EmbeddingModel:
    """Get the current default embedding model.

    Returns
    -------
    EmbeddingModel
        A model built from :envvar:`GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL`,
        or text-embedding-3-small by default.
    """
    return EmbeddingModel(model=get_settings().default_embedding_model)
