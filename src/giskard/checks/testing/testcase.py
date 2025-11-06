from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.check import Check
from ..core.interaction import Interaction
from ..generators import InteractionGenerator

if TYPE_CHECKING:
    from ..testing.runner import TestCaseResult


"""Test case model and runner integration.

`TestCase` binds a specific interaction or interaction generator with a sequence of `Check`s
and delegates execution to a `TestRunner`. It offers a single `run()` method
that returns a `TestCaseResult` summarizing the outcomes.
"""


class TestCase(BaseModel):
    """Bundle an interaction or interaction generator with a set of checks to execute.

    Attributes
    ----------
    name:
        Optional label for the test case.
    interaction:
        The interaction or interaction generator that produces interactions for testing.
        If an `Interaction` is provided, it will be used directly. If an `InteractionGenerator`
        is provided, it will be called to generate the interaction.
    checks:
        Sequence of checks to run against the generated interaction.
    """

    # Prevent pytest from attempting to collect this class as a test
    __test__ = False

    name: str | None = Field(None, description="Test case name")
    interaction: Interaction[Any, Any] | InteractionGenerator = Field(
        ..., description="Interaction or interaction generator"
    )
    checks: Sequence[Check] = Field(..., description="Test case checks")

    async def run(self, max_runs: int = 1) -> TestCaseResult:
        """Execute the test case using the configured `TestRunner`.

        Parameters
        ----------
        max_runs : int, default=1
            Number of times to generate interactions and run checks.
            When max_runs=1, generates a single interaction and runs all checks against it.
            When max_runs>1, generates fresh interactions for each run and stops
            at the first failed check.

        Returns
        -------
        TestCaseResult
            Results aggregated across all runs, with total_runs indicating
            how many runs were actually executed (may be less than max_runs
            if stopped early due to failure).
        """
        # Lazy import to avoid circular dependency with runner importing TestCase
        from ..testing.runner import get_runner

        runner = get_runner()
        return await runner.run(self, max_runs)

    async def assert_passed(self, max_runs: int = 1) -> None:
        """Run the test case and assert that it passed.

        This is a convenience method that combines running the test case with
        asserting that it passed. It's equivalent to:

        ```python
        result = await test_case.run(max_runs)
        result.assert_passed()
        ```

        Parameters
        ----------
        max_runs : int, default=1
            Number of times to generate interactions and run checks.
            When max_runs=1, generates a single interaction and runs all checks against it.
            When max_runs>1, generates fresh interactions for each run and stops
            at the first failed check.

        Raises
        ------
        AssertionError
            If the test case did not pass, with formatted failure messages as the error message.
        """
        result = await self.run(max_runs)
        result.assert_passed()
