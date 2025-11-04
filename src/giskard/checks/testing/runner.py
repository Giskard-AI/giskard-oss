from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..core.check import Check, CheckResult
from ..core.context import Context
from ..core.interaction_result import InteractionResult

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime import cycle
    from ..testing.testcase import TestCase


"""Test runner and results aggregation.

This module contains a minimal `TestRunner` that executes checks, captures
exceptions as error results, measures durations, and aggregates everything into
an immutable `TestCaseResult` with convenience properties.
"""


class TestCaseResult(BaseModel):
    """Immutable summary of a test case execution with full run history."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    all_runs: list[list[CheckResult]]  # Primary data: all runs
    duration_ms: int
    total_runs: int = 1

    @property
    def results(self) -> list[CheckResult]:
        """Final run results."""
        return self.all_runs[-1] if self.all_runs else []

    @property
    def passed(self) -> bool:
        """True when all checks passed."""
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> bool:
        """True when at least one check failed and none errored."""
        return not self.errored and any(result.failed for result in self.results)

    @property
    def errored(self) -> bool:
        """True when at least one check errored."""
        return any(result.errored for result in self.results)

    @property
    def skipped(self) -> bool:
        """True when all checks were skipped."""
        return all(result.skipped for result in self.results) and len(self.results) > 0

    def format_failures(self) -> list[str]:
        """Format failed check results into a list of readable error messages.

        Returns
        -------
        list[str]
            List of formatted error messages for failed checks. Each message includes
            the check name/kind and the failure reason.
        """
        failure_messages: list[str] = []
        for result in self.results:
            if result.failed or result.errored:
                check_name: str = result.details.get(
                    "check_name"
                ) or result.details.get("check_kind", "Unknown check")
                status = "ERRORED" if result.errored else "FAILED"
                message = result.message or "No specific error message provided"
                failure_messages.append(f"{check_name} {status}: {message}")
        return failure_messages

    def assert_passed(self) -> None:
        """Assert that the test case passed, raising an AssertionError with formatted failure messages if not.

        This is a convenience method for test code that combines the assertion check
        with formatted error reporting. It's equivalent to:

        ```python
        assert result.passed, result.format_failures()
        ```

        Raises
        ------
        AssertionError
            If the test case did not pass, with formatted failure messages as the error message.
        """
        if not self.passed:
            failure_messages = self.format_failures()
            error_msg = "Test case failed with the following errors:\n" + "\n".join(
                failure_messages
            )
            raise AssertionError(error_msg)


class TestRunner:
    """Execute checks for a `TestCase` and produce a `TestCaseResult`.

    The TestRunner is responsible for:
    - Executing checks sequentially against a test case's interaction
    - Capturing exceptions and converting them to error results
    - Measuring execution durations for both individual checks and the total test case
    - Aggregating all results into an immutable TestCaseResult

    The runner automatically adds timing and metadata to each check result's details,
    including the check's kind, name, and description for observability.
    """

    async def _run(
        self, interaction_result: InteractionResult[Any, Any], checks: Sequence[Check]
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

        for chk in checks:
            check_start_time = time.perf_counter()
            res: CheckResult | None = None

            try:
                res = await chk.run(interaction_result)
            except Exception as e:
                res = CheckResult.error(
                    message=f"Check '{chk.name or chk.kind}' failed with error: {str(e)}",
                    details={
                        "traceback": traceback.format_exc(),
                        "check_name": chk.name,
                        "check_kind": chk.kind,
                        "check_description": chk.description,
                        "exception_type": type(e).__name__,
                    },
                )

            # Update the result with the duration in details for observability
            res = res.model_copy(
                update={
                    "details": {
                        **(res.details or {}),
                        "duration_ms": int(
                            (time.perf_counter() - check_start_time) * 1000
                        ),
                        "check_kind": chk.kind,
                        "check_name": chk.name,
                        "check_description": chk.description,
                    }
                }
            )

            results.append(res)

        return results

    async def run(self, tc: "TestCase[Any]", max_runs: int = 1) -> TestCaseResult:
        start_time = time.perf_counter()

        all_runs: list[list[CheckResult]] = []

        for _ in range(max_runs):
            interaction_result = await tc.interaction.generate(Context())
            results = await self._run(interaction_result, tc.checks)
            all_runs.append(results)

            # Check if this run failed - if so, we can stop early
            run_passed = all(result.passed for result in results)
            if not run_passed:
                break

        end_time = time.perf_counter()
        total_duration_ms = int((end_time - start_time) * 1000)

        if not all_runs:
            raise ValueError("max_runs should be greater than 0")

        return TestCaseResult(
            all_runs=all_runs, duration_ms=total_duration_ms, total_runs=len(all_runs)
        )


_default_runner = TestRunner()


def get_runner() -> TestRunner:
    """Return the default process-wide `TestRunner` instance."""
    return _default_runner
