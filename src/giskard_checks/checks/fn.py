from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult, CheckSeverity
from giskard_checks.core.interactions import Interaction

"""Function-backed check implementation.

This module provides `FnCheck`, a concrete `Check` implementation that delegates
its logic to a user-provided callable, and a convenience `from_fn` factory to
instantiate it.

The callable can be synchronous or asynchronous and must return either:
- a `bool`: True -> success, False -> failure, or
- a `CheckResult`: used as-is
"""

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class FnCheck(Check[InteractionT]):
    """A `Check` whose logic is a Python callable.

    Parameters are modeled as pydantic fields. At runtime, the `run` method will
    invoke `fn` with the provided interaction and translate the result into a
    `CheckResult` when a boolean is returned.

    Note: The `fn` field is not serializable and will not be included in
    serialization. As a result, `FnCheck` instances cannot be reliably
    serialized/deserialized. This is intended for programmatic/test use only.
    """

    KIND: ClassVar[str | None] = "fn"

    fn: Callable[[InteractionT], Awaitable[bool | CheckResult] | bool | CheckResult] = (
        Field(
            exclude=True,
            repr=False,
            description="Function to execute for the check. Not serializable.",
        )
    )
    success_message: str | None = None
    failure_message: str | None = None
    severity: CheckSeverity = CheckSeverity.ERROR
    details: dict[str, Any] = Field(default_factory=dict)

    async def run(self, interaction: InteractionT) -> CheckResult:
        """Execute the function and normalize its result to a `CheckResult`."""
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
    """Create an `FnCheck` from a callable.

    Example
    -------
    ```python
    from giskard_checks.checks import from_fn

    chk = from_fn(lambda inter: inter.output is not None, name="has_output")
    ```
    """
    return FnCheck(
        name=name,
        description=description,
        fn=fn,
        severity=severity,
        success_message=success_message,
        failure_message=failure_message,
        details={} if details is None else details,
    )
