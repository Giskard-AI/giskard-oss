"""Test case model and runner integration.

`TestCase` binds a concrete `Trace` with a sequence of `Check`s and delegates
execution to a `TestCaseRunner`. It offers a single `run()` method that returns a
`TestCaseResult` summarizing the outcomes.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .check import Check
from .result import TestCaseResult
from .trace import Trace


class TestCase[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """Bundle a trace with a set of checks to execute.

    **Note**: For most use cases, the fluent API (`scenario().interact().check()`) is
    recommended as it's simpler and more readable. This class is useful for advanced
    use cases where you need direct control over trace construction and check execution.

    Attributes
    ----------
    name:
        Optional label for the test case.
    trace:
        The trace containing interactions to test against.
    checks:
        Sequence of checks to run against the trace.
    """

    # Prevent pytest from attempting to collect this class as a test
    __test__: bool = False

    name: str | None = Field(default=None, description="Test case name")
    trace: TraceType = Field(..., description="Trace to execute checks against")
    checks: Sequence[Check[InputType, OutputType, TraceType]] = Field(
        ..., description="Test case checks"
    )

    async def run(self, return_exception: bool = False) -> TestCaseResult:
        # Lazy import to avoid circular dependency with runner importing TestCase
        from ..testing.runner import get_runner

        runner = get_runner()
        return await runner.run(self, return_exception)

    async def assert_passed(self) -> None:
        """Run the test case and assert that it passed.

        This is a convenience method that combines running the test case with
        asserting that it passed. It's equivalent to:

        ```python
        result = await test_case.run(return_exception)
        result.assert_passed()
        ```

        Raises
        ------
        AssertionError
            If the test case did not pass, with formatted failure messages as the error message.
        """
        result = await self.run()
        result.assert_passed()
