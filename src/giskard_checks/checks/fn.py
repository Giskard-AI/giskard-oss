from __future__ import annotations

import inspect
from collections.abc import Awaitable, Iterable
from typing import Any, Callable, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult, CheckSeverity
from giskard_checks.core.extraction import Extractor, JsonPathExtractor
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


class EqualityCheck(Check[InteractionT]):
    KIND = "equality"

    expected: Any
    # Optional new extractor strategy (preferred)
    extractor: Extractor | None = Field(
        default=None, description="Optional extractor for selecting values"
    )
    # Backcompat convenience to select values via JSONPath when no extractor is provided
    key: str | None = Field(default=None, description="JSON path to the key to check")
    severity: CheckSeverity = CheckSeverity.ERROR

    async def run(self, interaction: InteractionT) -> CheckResult:
        # Determine values to compare
        if self.extractor is not None:
            values = self.extractor.extract(interaction)
        elif self.key:
            values = JsonPathExtractor(key=self.key).extract(interaction)
        else:
            # Default to selecting the output field via JSONPath
            values = JsonPathExtractor(key="output").extract(interaction)

        matched = any(value == self.expected for value in values)
        if matched:
            return CheckResult.success(
                message=f"Input is equal to {self.expected}", details={}
            )

        return CheckResult.failure(
            message=f"Input is not equal to {self.expected}",
            details={
                "actual": values,
                "expected": self.expected,
                "key": self.key,
            },
        )


# TODO: check DataInContext return type from jsonpath_expr.find
class StringMatchingCheck(Check[InteractionT]):
    KIND: ClassVar[str | None] = "string_matching"

    content: str = Field(..., description="The string to match in the output")

    # TODO: case_sensitive: bool = Field(default=True, description="Whether the string matching should be case sensitive")

    # Optional new extractor strategy (preferred)
    extractor: Extractor | None = Field(
        default=None, description="Optional extractor for selecting values"
    )
    # Backcompat convenience to select values via JSONPath when no extractor is provided
    key: str | None = Field(default=None, description="JSON path to the key to check")
    match_all: bool = Field(
        default=False, description="Whether all strings should match"
    )

    async def run(self, interaction: InteractionT) -> CheckResult:
        # Extract values using configured extractor or fall back to JSONPath/raw output
        if self.extractor is not None:
            values = self.extractor.extract(interaction)
        elif self.key:
            values = JsonPathExtractor(key=self.key).extract(interaction)
        else:
            # Default to selecting the output field via JSONPath
            values = JsonPathExtractor(key="output").extract(interaction)

        texts: list[str] = []
        for item in values:
            value = getattr(item, "value", item)
            texts.append(value if isinstance(value, str) else str(value))

        if self.match_all:
            matched = all(self.content in text for text in texts)
        else:
            matched = any(self.content in text for text in texts)

        if matched:
            return CheckResult.success(message=f"String matching succeeded", details={})
        else:
            return CheckResult.failure(message=f"String matching failed", details={})


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
                    message=self.success_message,
                    details=self.details,
                )
            return CheckResult.failure(
                message=self.failure_message,
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
