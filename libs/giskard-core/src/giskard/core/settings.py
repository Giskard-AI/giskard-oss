"""Environment-backed settings for giskard-core."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .environment import is_truthy_env


class GiskardCoreSettings(BaseSettings):
    """Environment-backed settings for giskard-core.

    Values can be set via environment variables prefixed with ``GISKARD_``
    or in a ``.env`` file at the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="GISKARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hide_welcome: bool = Field(
        default=False,
        description="Suppress the one-shot enterprise welcome message on suite/scan runs.",
    )

    @field_validator("hide_welcome", mode="before")
    @classmethod
    def _normalize_hide_welcome(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return is_truthy_env(str(value))


def get_settings() -> GiskardCoreSettings:
    """Return settings loaded from the environment."""
    return GiskardCoreSettings()
