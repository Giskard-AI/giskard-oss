from __future__ import annotations

import os
import warnings
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, computed_field, model_validator

from .interactions import Interaction

"""Core checking primitives.

This module defines the foundational types used by the library to represent
checks, their execution results, and related enums. The key abstractions are:

- `CheckStatus`: outcome categories for a single check execution
- `CheckSeverity`: importance of a check when it fails
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


class CheckSeverity(str, Enum):
    """Represents how important a check is when it fails."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckResult(BaseModel):
    """Immutable result produced by running a `Check`.

    Attributes
    ----------
    kind:
        Check type identifier (usually provided by the `Check` subclass `KIND`).
    name:
        Human-friendly name of the check.
    description:
        Optional description of what the check validates.
    status:
        Outcome status of the check.
    message:
        Optional short message to surface to users (e.g., success/failure reason).
    traceback:
        Captured traceback if the check raised an unhandled exception.
    duration_ms:
        Execution time in milliseconds.
    severity:
        Importance of the check when it fails.
    details:
        Arbitrary structured payload with additional context.
    """

    model_config = {"frozen": True}
    kind: str = Field(..., description="Check type identifier")
    name: str | None = Field(None, description="Check name")
    description: str | None = Field(None, description="Check description")
    status: CheckStatus = Field(..., description="Check status")
    message: str | None = Field(None, description="Check message")
    traceback: str | None = Field(None, description="Check traceback")
    duration_ms: int | None = Field(None, description="Check duration in milliseconds")
    severity: CheckSeverity = Field(..., description="Check severity")
    details: dict[str, Any] = Field(default_factory=dict, description="Check details")

    # Convenience constructors
    @classmethod
    def success(
        cls,
        *,
        kind: str,
        name: str | None = None,
        description: str | None = None,
        message: str | None = None,
        severity: CheckSeverity = CheckSeverity.INFO,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a successful result.

        Parameters mirror the fields on the model. `severity` defaults to
        `INFO` for a success, and `details` is normalized to an empty map if
        not provided.
        """
        return cls(
            kind=kind,
            name=name,
            description=description,
            status=CheckStatus.PASS,
            message=message,
            traceback=None,
            duration_ms=None,
            severity=severity,
            details={} if details is None else details,
        )

    @classmethod
    def failure(
        cls,
        *,
        kind: str,
        name: str | None = None,
        description: str | None = None,
        message: str | None = None,
        severity: CheckSeverity = CheckSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a failure result."""
        return cls(
            kind=kind,
            name=name,
            description=description,
            status=CheckStatus.FAIL,
            message=message,
            traceback=None,
            duration_ms=None,
            severity=severity,
            details={} if details is None else details,
        )

    @classmethod
    def skip(
        cls,
        *,
        kind: str,
        name: str | None = None,
        description: str | None = None,
        message: str | None = None,
        severity: CheckSeverity = CheckSeverity.INFO,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a skipped result (e.g., precondition not met)."""
        return cls(
            kind=kind,
            name=name,
            description=description,
            status=CheckStatus.SKIP,
            message=message,
            traceback=None,
            duration_ms=None,
            severity=severity,
            details={} if details is None else details,
        )

    @classmethod
    def error(
        cls,
        *,
        kind: str,
        name: str | None = None,
        description: str | None = None,
        message: str | None = None,
        traceback: str | None = None,
        severity: CheckSeverity = CheckSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct an error result from an exception or unexpected condition."""
        return cls(
            kind=kind,
            name=name,
            description=description,
            status=CheckStatus.ERROR,
            message=message,
            traceback=traceback,
            duration_ms=None,
            severity=severity,
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
    KIND: ClassVar[str | None] = Field(
        default=None, description="Check type identifier"
    )
    name: str | None = Field(default=None, description="Check name")
    description: str | None = Field(default=None, description="Check description")
    params: dict[str, Any] = Field(default_factory=dict)

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

        class_kind = getattr(cls, "KIND", None)
        if isinstance(class_kind, str) and class_kind:
            _register_check_kind(class_kind, cls)

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
    def from_dict(cls, data: dict[str, Any]) -> "Check[Any]":
        """Instantiate a concrete `Check` from serialized data.

        Expects a `kind` field to resolve the target subclass using the global
        registry. Extra fields not defined on the subclass are ignored by default.
        """
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Serialized check must include non-empty 'kind'")
        target_cls = _CHECK_KIND_REGISTRY.get(kind)
        if target_cls is None:
            raise ValueError(f"Unknown check kind '{kind}'; is the class imported?")
        return target_cls.model_validate(data)

    async def run(self, interaction: InteractionT) -> CheckResult:
        """Execute the check against the provided interaction.

        Subclasses must override this method and return a `CheckResult`. The
        implementation may be async.
        """
        raise NotImplementedError


# Optional global registry for Check kinds
_CHECK_KIND_REGISTRY: dict[str, type["Check[Any]"]] = {}
_ENFORCE_KIND_UNIQUENESS: bool = os.getenv(
    "GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS", "1"
).lower() in {"1", "true", "yes", "on"}


def _register_check_kind(kind: str, cls: type["Check[Any]"]) -> None:
    """Register a check class for a given kind.

    If `GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS` is truthy (default), a duplicate
    registration with a different class raises a `ValueError`. Otherwise a
    warning is emitted and the new class overwrites the previous one.
    """
    existing = _CHECK_KIND_REGISTRY.get(kind)
    if existing is not None and existing is not cls:
        if _ENFORCE_KIND_UNIQUENESS:
            raise ValueError(
                f"Duplicate check KIND '{kind}' for classes {existing.__name__} and {cls.__name__}"
            )
        warnings.warn(
            f"Duplicate check KIND '{kind}' detected for {existing.__name__} and {cls.__name__}; "
            "latest class will overwrite the registry entry.",
            RuntimeWarning,
            stacklevel=2,
        )
    _CHECK_KIND_REGISTRY[kind] = cls
