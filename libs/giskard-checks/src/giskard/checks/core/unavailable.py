"""Placeholder for Giskard Hub checks that are not available in this environment."""

import warnings
from pathlib import Path
from typing import Any

import giskard.core
import pydantic
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

from .check import Check
from .interaction import Trace
from .result import CheckResult

HUB_KIND_PREFIX = "hub_"
"""Kind prefix reserved for checks provided by Giskard Hub."""

# Attribute the warning to the caller that loaded the payload, not to the
# validation machinery it went through.
_WARNING_SKIP_PREFIXES = tuple(
    str(Path(module.__file__).parent)
    for module in (pydantic, giskard.core)
    if module.__file__ is not None
) + (str(Path(__file__).parents[1]),)


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
    """

    unavailable_kind: str = Field(description="Kind of the unavailable Hub check")
    spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Original check configuration, without kind, name or description",
    )

    @computed_field
    def kind(self) -> str | None:
        """The original Hub kind, so dumps and results keep reporting it."""
        return self.unavailable_kind

    @classmethod
    def from_payload(cls, kind: str, value: dict[str, Any]) -> "UnavailableHubCheck":
        """Build the placeholder from a raw check payload and warn about it."""
        spec = {
            k: v for k, v in value.items() if k not in {"kind", "name", "description"}
        }
        warnings.warn(
            f"Check kind '{kind}' is provided by Giskard Hub and is not available "
            "in this environment; it will be skipped. Run it on Giskard Hub.",
            UnavailableCheckWarning,
            skip_file_prefixes=_WARNING_SKIP_PREFIXES,
        )
        return cls(
            name=value.get("name"),
            description=value.get("description"),
            unavailable_kind=kind,
            spec=spec,
        )

    @model_serializer(mode="wrap")
    def _serialize_as_original(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        data.pop("unavailable_kind", None)
        spec: dict[str, Any] = data.pop("spec", {})
        return {**data, **spec}

    async def run(self, trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.skip(
            message=(
                f"Check kind '{self.unavailable_kind}' is provided by Giskard Hub "
                "and cannot run in this environment."
            )
        )
