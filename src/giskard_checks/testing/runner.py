from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from giskard_checks.core.check import CheckResult

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime import cycle
    from giskard_checks.testing.testcase import TestCase


"""Test runner and results aggregation.

This module contains a minimal `TestRunner` that executes checks, captures
exceptions as error results, measures durations, and aggregates everything into
an immutable `TestCaseResult` with convenience properties.
"""


class TestCaseResult(BaseModel):
    """Immutable summary of a test case execution."""

    model_config = {"frozen": True}
    results: list[CheckResult]
    duration_ms: int

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

    async def run(self, tc: "TestCase[Any]") -> TestCaseResult:
        results: list[CheckResult] = []

        start_time = time.perf_counter()
        for chk in tc.checks:
            check_start_time = time.perf_counter()
            res: CheckResult | None = None
            try:
                res = await chk.run(tc.interaction)
            except Exception as e:
                res = CheckResult.error(
                    message=str(e),
                    details={"traceback": traceback.format_exc()},
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

        end_time = time.perf_counter()
        total_duration_ms = int((end_time - start_time) * 1000)

        return TestCaseResult(results=results, duration_ms=total_duration_ms)


_default_runner = TestRunner()


def get_runner() -> TestRunner:
    """Return the default process-wide `TestRunner` instance."""
    return _default_runner
