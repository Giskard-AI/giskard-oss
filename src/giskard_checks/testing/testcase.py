from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SkipValidation, field_serializer, field_validator

from giskard_checks.core.check import Check
from giskard_checks.core.interactions import Interaction

if TYPE_CHECKING:
    from giskard_checks.testing.runner import TestCaseResult


"""Test case model and runner integration.

`TestCase` binds a specific `Interaction` instance with a sequence of `Check`s
and delegates execution to a `TestRunner`. It offers a single `run()` method
that returns a `TestCaseResult` summarizing the outcomes.
"""


class TestCase(BaseModel):
    """Bundle a single interaction with a set of checks to execute.

    Attributes
    ----------
    name:
        Optional label for the test case.
    interaction:
        The interaction under test.
    checks:
        Sequence of checks to run against the interaction.
    """

    # Prevent pytest from attempting to collect this class as a test
    __test__ = False

    name: str | None = Field(None, description="Test case name")
    # Validation is skipped for the interaction field to allow for generic deserialization
    interaction: Interaction[Any, Any] = Field(..., description="Test case interaction")
    checks: Sequence[Check] = Field(..., description="Test case checks")

    async def run(self) -> TestCaseResult:
        """Execute the test case using the configured `TestRunner`."""
        # Lazy import to avoid circular dependency with runner importing TestCase
        from giskard_checks.testing.runner import get_runner

        runner = get_runner()
        return await runner.run(self)

    async def assert_passed(self) -> None:
        """Run the test case and assert that it passed.

        This is a convenience method that combines running the test case with
        asserting that it passed. It's equivalent to:

        ```python
        result = await test_case.run()
        result.assert_passed()
        ```

        Raises
        ------
        AssertionError
            If the test case did not pass, with formatted failure messages as the error message.
        """
        result = await self.run()
        result.assert_passed()
