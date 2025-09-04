from __future__ import annotations

import os
import warnings
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

from .interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class CheckSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckResult(BaseModel):
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
        return self.status == CheckStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == CheckStatus.FAIL

    @property
    def errored(self) -> bool:
        return self.status == CheckStatus.ERROR

    @property
    def skipped(self) -> bool:
        return self.status == CheckStatus.SKIP


class Check(BaseModel, Generic[InteractionT]):
    # Preferred usage: set a class-level KIND on subclasses, e.g. KIND = "my_check".
    # This removes the need to pass `kind` to each instance and allows registry validation.
    KIND: ClassVar[str | None] = Field(
        default=None, description="Check type identifier"
    )
    name: str | None = Field(None, description="Check name")
    description: str | None = Field(None, description="Check description")
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

    @property
    def kind(self) -> str:
        class_kind = getattr(self.__class__, "KIND", None)
        if not (isinstance(class_kind, str) and class_kind):
            raise ValueError(f"KIND must be set for {self.__class__.__name__}")
        return class_kind

    async def run(self, interaction: InteractionT) -> CheckResult:
        raise NotImplementedError


# Optional global registry for Check kinds
_CHECK_KIND_REGISTRY: dict[str, type["Check[Any]"]] = {}
_ENFORCE_KIND_UNIQUENESS: bool = os.getenv(
    "GISKARD_CHECK_KIND_ENFORCE_UNIQUENESS", "1"
).lower() in {"1", "true", "yes", "on"}


def _register_check_kind(kind: str, cls: type["Check[Any]"]) -> None:
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
