"""Placeholder for Giskard Hub checks that are not available in this environment."""

import os
import warnings
from typing import Any

import giskard.core
import pydantic
from pydantic import ConfigDict, Field, computed_field

from .check import Check
from .interaction import Trace
from .result import CheckResult

HUB_KIND_PREFIX = "hub_"
"""Kind prefix reserved for checks provided by Giskard Hub."""

# Attribute the warning to the caller that loaded the payload, not to the
# validation machinery it went through. The trailing separator keeps e.g. the
# ``pydantic`` prefix from also matching ``pydantic_settings``.
_WARNING_SKIP_PREFIXES = tuple(
    os.path.join(os.path.dirname(path), "")
    for path in (pydantic.__file__, giskard.core.__file__, os.path.dirname(__file__))
)


class UnavailableCheckWarning(UserWarning):
    """Emitted when a loaded check cannot run in the current environment.

    Environments that are expected to provide every Hub check (such as the Hub
    itself) can turn this into an error with
    ``warnings.simplefilter("error", UnavailableCheckWarning)``.
    """


class UnavailableHubCheck(Check[Any, Any, Trace[Any, Any]]):
    """Stand-in for a Giskard Hub check whose kind is not registered here.

    Hub checks (kinds prefixed with ``hub_``) are only registered when the Hub
    runtime is imported. Loading one elsewhere yields this placeholder instead
    of a validation error: running it returns a skipped result, and it
    serializes back to the original payload so the spec round-trips unchanged.

    This class is intentionally not registered under any kind.

    Notes
    -----
    This is the one ``Check`` that sets ``extra="allow"``, despite the rule in
    ``Discriminated``. That rule guards against unknown keys being silently
    dropped; here they are kept verbatim as extra fields, so pydantic dumps
    them back at the top level with every dump option applied.
    """

    model_config = ConfigDict(extra="allow")

    unavailable_kind: str = Field(
        exclude=True, description="Kind of the unavailable Hub check"
    )

    @computed_field
    def kind(self) -> str:
        """The original Hub kind, so dumps and results keep reporting it."""
        return self.unavailable_kind

    @classmethod
    def from_payload(cls, kind: str, value: dict[str, Any]) -> "UnavailableHubCheck":
        """Build the placeholder from a raw check payload and warn about it."""
        warnings.warn(
            f"Check kind '{kind}' is provided by Giskard Hub and is not available "
            "in this environment; it will be skipped. Run it on Giskard Hub.",
            UnavailableCheckWarning,
            skip_file_prefixes=_WARNING_SKIP_PREFIXES,
        )
        return cls.model_validate({**value, "unavailable_kind": kind})

    async def run(self, trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.skip(
            message=(
                f"Check kind '{self.unavailable_kind}' is provided by Giskard Hub "
                "and cannot run in this environment."
            )
        )
