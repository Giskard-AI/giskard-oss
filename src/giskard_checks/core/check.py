from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..utils.discriminated import Discriminated, discriminated_base
from .interaction_result import InteractionResult

"""Core checking primitives.

This module defines the foundational types used by the library to represent
checks, their execution results, and related enums. The key abstractions are:

- `CheckStatus`: outcome categories for a single check execution
- `CheckResult`: immutable record describing the outcome of a check
- `Check`: base class to implement concrete checks

It also provides a small global registry keyed by `Check.KIND` to help detect
duplicate kinds during development. Set the environment variable
`GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` to control enforcement (enabled by
default).
"""


class CheckStatus(str, Enum):
    """Outcome categories for a check execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class Metric(BaseModel):
    """A named metric value captured during check execution.

    Metrics provide a way to attach quantitative measurements to check results,
    such as performance timings, confidence scores, or other numerical values
    that provide additional context about the check execution.

    Attributes
    ----------
    name : str
        The name/identifier of the metric
    value : float
        The numerical value of the metric
    """

    name: str
    value: float


class CheckResult(BaseModel):
    """Immutable result produced by running a `Check`.

    Attributes
    ----------
    status:
        Outcome status of the check.
    message:
        Optional short message to surface to users (e.g., success/failure reason).
    metrics:
        List of auxiliary metrics captured by the check.
    details:
        Arbitrary structured payload with additional context (e.g., failure reasons,
        timings, and any metadata the check wishes to include).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    status: CheckStatus = Field(..., description="Check status")
    message: str | None = Field(default=None, description="Check message")
    metrics: list[Metric] = Field(default_factory=list, description="Check metric")
    details: dict[str, Any] = Field(default_factory=dict, description="Check details")

    # Convenience constructors
    @classmethod
    def success(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a successful result.

        Parameters mirror the fields on the model. `details` is normalized to
        an empty map if not provided.
        """
        return cls(
            status=CheckStatus.PASS,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def failure(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a failure result."""
        return cls(
            status=CheckStatus.FAIL,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def skip(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a skipped result (e.g., precondition not met)."""
        return cls(
            status=CheckStatus.SKIP,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def error(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct an error result from an exception or unexpected condition."""
        return cls(
            status=CheckStatus.ERROR,
            message=message,
            details={} if details is None else details,
        )

    @property
    def passed(self) -> bool:
        """Return True if `status` is `PASS`."""
        return self.status == CheckStatus.PASS

    @property
    def failed(self) -> bool:
        """Return True if `status` is `FAIL`."""
        return self.status == CheckStatus.FAIL

    @property
    def errored(self) -> bool:
        """Return True if `status` is `ERROR`."""
        return self.status == CheckStatus.ERROR

    @property
    def skipped(self) -> bool:
        """Return True if `status` is `SKIP`."""
        return self.status == CheckStatus.SKIP


@discriminated_base
class Check(Discriminated):
    """Base class for checks.

    Subclasses should be registered using the @Check.register("kind") decorator
    to enable polymorphic serialization and deserialization.
    """

    name: str | None = Field(default=None, description="Check name")
    description: str | None = Field(default=None, description="Check description")

    async def run(self, interaction: InteractionResult[Any, Any]) -> CheckResult:
        """Execute the check against the provided interaction result.

        Subclasses must override this method and return a `CheckResult`. The
        implementation may be async.

        Parameters
        ----------
        interaction : InteractionResult[Any, Any]
            The interaction result to check against
        """
        raise NotImplementedError
