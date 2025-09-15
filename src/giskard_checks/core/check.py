from __future__ import annotations

import os
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from .interactions import Interaction
from .registry import Registry

"""Core checking primitives.

This module defines the foundational types used by the library to represent
checks, their execution results, and related enums. The key abstractions are:

- `CheckStatus`: outcome categories for a single check execution
- `CheckResult`: immutable record describing the outcome of a check
- `Check`: generic base class to implement concrete checks

It also provides a small global registry keyed by `Check.KIND` to help detect
duplicate kinds during development. Set the environment variable
`GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` to control enforcement (enabled by
default).
"""

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


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


class Check(BaseModel, Generic[InteractionT]):
    """Base class for checks.

    Subclasses must define a non-empty class attribute `KIND` (e.g.,
    `KIND = "my_check"`). The provided model validator ensures that instances
    are consistent with the declared kind and enables an optional global
    registry of kinds to detect accidental duplicates.
    """

    # Preferred usage: set a class-level KIND on subclasses, e.g. KIND = "my_check".
    # This removes the need to pass `kind` to each instance and allows registry validation.
    KIND: ClassVar[str | None] = None
    name: str | None = Field(default=None, description="Check name")
    description: str | None = Field(default=None, description="Check description")

    @model_validator(mode="before")
    @classmethod
    def _populate_and_validate_kind(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        provided = data.get("kind")
        class_kind = getattr(cls, "KIND", None)
        if not (isinstance(class_kind, str) and class_kind):
            raise ValueError(f"KIND must be set for {cls.__name__}")
        if provided is not None and provided != class_kind:
            raise ValueError(
                f"kind '{provided}' does not match class KIND '{class_kind}' for {cls.__name__}"
            )
        return data

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip the exact base class and any abstract intermediates
        if cls is Check or getattr(cls, "__abstractmethods__", None):
            return

        # Skip generic instantiations (like Check[SomeInteraction])
        # Only register the actual class definitions
        if (
            hasattr(cls, "__origin__")
            or getattr(cls, "__args__", None)
            or "[" in cls.__name__
        ):
            return

        class_kind = getattr(cls, "KIND", None)
        if isinstance(class_kind, str) and class_kind:
            _CHECK_REGISTRY.register(class_kind, cls)

    @computed_field(return_type=str)
    @property
    def kind(self) -> str:
        """Return the check type identifier from the class `KIND`.

        Marked as a computed field so it's included in serialization via
        `model_dump()` and JSON dumps, enabling generic deserialization.
        """
        class_kind = getattr(self.__class__, "KIND", None)
        if not (isinstance(class_kind, str) and class_kind):
            raise ValueError(f"KIND must be set for {self.__class__.__name__}")
        return class_kind

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "Check[Any]":
        """Instantiate a concrete `Check` from serialized data.

        Uses registry-based format with `kind` field for deserialization.
        """
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Serialized check must include a non-empty 'kind' field")

        target_cls = _CHECK_REGISTRY.get_or_raise(kind)
        return target_cls.model_validate(data)

    def serialize(self) -> dict[str, Any]:
        """Serialize the check into a JSON-friendly dict.

        The output includes the computed `kind` field to enable registry-based
        deserialization.
        """
        return self.model_dump()

    async def run(self, interaction: InteractionT) -> CheckResult:
        """Execute the check against the provided interaction.

        Subclasses must override this method and return a `CheckResult`. The
        implementation may be async.

        Parameters
        ----------
        interaction : InteractionT
            The interaction to check against
        """
        raise NotImplementedError


# Global registry for Check kinds
_ENFORCE_KIND_UNIQUENESS: bool = os.getenv(
    "GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS", "1"
).lower() in {"1", "true", "yes", "on"}

_CHECK_REGISTRY = Registry[Check[Any]](
    name="check", enforce_uniqueness=_ENFORCE_KIND_UNIQUENESS
)


def list_registered_check_kinds() -> list[str]:
    """List all registered check kinds in alphabetical order.

    Returns
    -------
    list[str]
        Sorted list of all registered check kind identifiers
    """
    return _CHECK_REGISTRY.list_kinds()
