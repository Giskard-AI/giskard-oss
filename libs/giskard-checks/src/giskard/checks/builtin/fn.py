from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Callable, override

from pydantic import Field

from ..core.check import Check
from ..core.result import CheckResult
from ..core.trace import Trace

"""Function-backed check implementation.

This module provides `FnCheck`, a concrete `Check` implementation that delegates
its logic to a user-provided callable, and a convenience `from_fn` factory to
instantiate it.

The callable can be synchronous or asynchronous and must return either:
- a `bool`: True -> success, False -> failure, or
- a `CheckResult`: used as-is
"""


@Check.register("fn")
class FnCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]@
    Check[InputType, OutputType, TraceType]
):  # pyright: ignore[reportMissingTypeArgument]
    """A `Check` whose logic is a Python callable.

    Parameters are modeled as pydantic fields. At runtime, the `run` method will
    invoke `fn` with the provided trace and translate the result into a
    `CheckResult` when a boolean is returned.

    Note: The `fn` field is not serializable and will not be included in
    serialization. As a result, `FnCheck` instances cannot be reliably
    serialized/deserialized. This is intended for programmatic/test use only.
    """

    fn: Callable[
        [TraceType],
        Awaitable[bool | CheckResult] | bool | CheckResult,
    ] = Field(
        exclude=True,
        repr=False,
        description="Function to execute for the check. Not serializable.",
    )
    success_message: str | None = None
    failure_message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the function and normalize its result to a `CheckResult`."""
        result = self.fn(trace)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, CheckResult):
            return result

        if isinstance(result, bool):
            if result:
                return CheckResult.success(
                    message=self.success_message,
                    details=self.details,
                )
            return CheckResult.failure(
                message=self.failure_message,
                details=self.details,
            )

        result_type = type(result).__name__
        raise TypeError(
            "from_fn callable must return bool or CheckResult (or awaitable thereof), "
            + f"but got {result_type} (value: {result!r})"
        )


def from_fn[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    fn: Callable[
        [TraceType],
        Awaitable[bool | CheckResult] | bool | CheckResult,
    ],
    *,
    name: str | None = None,
    description: str | None = None,
    success_message: str | None = None,
    failure_message: str | None = None,
    details: dict[str, Any] | None = None,
) -> Check[InputType, OutputType, TraceType]:
    """Create an `FnCheck` from a callable.

    Example
    -------
    ```python
    from giskard.checks import from_fn

    chk = from_fn(lambda trace: trace.last.outputs is not None, name="has_output")
    ```
    """
    return FnCheck(
        name=name,
        description=description,
        fn=fn,
        success_message=success_message,
        failure_message=failure_message,
        details={} if details is None else details,
    )
