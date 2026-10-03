"""Test case model and runner integration.

`TestCase` binds a concrete `Trace` with a sequence of `Check`s and delegates
execution to a `TestCaseRunner`. Its async `run()` and synchronous `run_sync()`
methods return a `TestCaseResult` summarizing the outcomes.
"""

from collections.abc import Sequence
from typing import Any, overload

from pydantic import BaseModel, Field

from ._run_sync import run_sync as _run_sync
from .check import Check
from .interaction import Trace
from .result import TestCaseResult


class TestCase[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """Bundle a trace with a set of checks to execute.

    **Note**: For most use cases, the fluent API (`Scenario(...).interact().check()`) is
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

    @overload
    def run_sync(self, return_exception: bool, /) -> TestCaseResult: ...

    @overload
    def run_sync(self, *, return_exception: bool = False) -> TestCaseResult: ...

    def run_sync(self, *args: Any, **kwargs: Any) -> TestCaseResult:
        """Execute the test case synchronously.

        Parameters
        ----------
        return_exception : bool, default False
            If True, return results when exceptions occur instead of raising.

        Returns
        -------
        TestCaseResult
            The test case execution result.

        Raises
        ------
        RuntimeError
            If called while an asyncio event loop is already running. In that
            case, use ``await test_case.run(...)`` instead.
        """
        return _run_sync(self.run, *args, **kwargs)

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
