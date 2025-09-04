from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, ClassVar, Generic, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult, CheckSeverity
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class FnCheck(Check[InteractionT]):
    KIND: ClassVar[str | None] = "fn"

    fn: Callable[[InteractionT], Awaitable[bool | CheckResult] | bool | CheckResult]
    success_message: str | None = None
    failure_message: str | None = None
    severity: CheckSeverity = CheckSeverity.ERROR
    details: dict[str, Any] = Field(default_factory=dict)

    async def run(self, interaction: InteractionT) -> CheckResult:
        result = self.fn(interaction)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, CheckResult):
            return result

        if isinstance(result, bool):
            if result:
                return CheckResult.success(
                    kind=self.kind,
                    name=self.name,
                    description=self.description,
                    message=self.success_message,
                    severity=self.severity,
                    details=self.details,
                )
            return CheckResult.failure(
                kind=self.kind,
                name=self.name,
                description=self.description,
                message=self.failure_message,
                severity=self.severity,
                details=self.details,
            )

        raise TypeError(
            "from_fn callable must return bool or CheckResult (or awaitable thereof)"
        )


def from_fn(
    fn: Callable[[InteractionT], Awaitable[bool | CheckResult] | bool | CheckResult],
    *,
    name: str | None = None,
    description: str | None = None,
    severity: CheckSeverity = CheckSeverity.ERROR,
    success_message: str | None = None,
    failure_message: str | None = None,
    details: dict[str, Any] | None = None,
) -> Check[InteractionT]:
    return FnCheck(
        name=name,
        description=description,
        fn=fn,
        severity=severity,
        success_message=success_message,
        failure_message=failure_message,
        details={} if details is None else details,
    )
