"""Environment-backed settings for giskard-core."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


def get_settings() -> GiskardCoreSettings:
    """Return settings loaded from the environment."""
    return GiskardCoreSettings()
