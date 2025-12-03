from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .trace import Trace


class CheckStatus(str, Enum):
    """Outcome categories for a check execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class Metric(BaseModel):
    """A named metric value captured during check execution.

    Metrics provide a way to attach quantitative measurements to check results,
    such as performance timings, confidence scores, or other numerical values
    that provide additional context about the check execution.

    Attributes
    ----------
    name : str
        The name/identifier of the metric
    value : float
        The numerical value of the metric
    """

    name: str
    value: float


class CheckResult(BaseModel):
    """Immutable result produced by running a `Check`.

    Attributes
    ----------
    status:
        Outcome status of the check.
    message:
        Optional short message to surface to users (e.g., success/failure reason).
    metrics:
        List of auxiliary metrics captured by the check.
    details:
        Arbitrary structured payload with additional context (e.g., failure reasons,
        timings, and any metadata the check wishes to include).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    status: CheckStatus = Field(..., description="Check status")
    message: str | None = Field(default=None, description="Check message")
    metrics: list[Metric] = Field(default_factory=list, description="Check metric")
    details: dict[str, Any] = Field(default_factory=dict, description="Check details")

    # Convenience constructors
    @classmethod
    def success(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a successful result.

        Parameters mirror the fields on the model. `details` is normalized to
        an empty map if not provided.
        """
        return cls(
            status=CheckStatus.PASS,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def failure(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a failure result."""
        return cls(
            status=CheckStatus.FAIL,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def skip(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct a skipped result (e.g., precondition not met)."""
        return cls(
            status=CheckStatus.SKIP,
            message=message,
            details={} if details is None else details,
        )

    @classmethod
    def error(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Construct an error result from an exception or unexpected condition."""
        return cls(
            status=CheckStatus.ERROR,
            message=message,
            details={} if details is None else details,
        )

    @property
    def passed(self) -> bool:
        """Return True if `status` is `PASS`."""
        return self.status == CheckStatus.PASS

    @property
    def failed(self) -> bool:
        """Return True if `status` is `FAIL`."""
        return self.status == CheckStatus.FAIL

    @property
    def errored(self) -> bool:
        """Return True if `status` is `ERROR`."""
        return self.status == CheckStatus.ERROR

    @property
    def skipped(self) -> bool:
        """Return True if `status` is `SKIP`."""
        return self.status == CheckStatus.SKIP


class ScenarioResult[InputType, OutputType](BaseModel):
    """Result of executing an entire scenario.

    Attributes
    ----------
    scenario_name:
        Name of the scenario that was executed.
    check_results:
        List of all check results from executed checks.
    passed:
        Whether all executed checks passed.
    duration_ms:
        Total execution time in milliseconds.
    final_trace:
        The trace state after execution, containing all interactions that occurred.
    """

    scenario_name: str = Field(..., description="Scenario name")
    steps: list[TestCaseResult]
    duration_ms: int = Field(..., description="Total execution time in milliseconds")
    final_trace: Trace[InputType, OutputType] = Field(
        ..., description="Final trace state after execution"
    )

    @property
    def passed(self) -> bool:
        """Whether all executed steps passed."""
        return all(step.passed for step in self.steps)

    @property
    def failed(self) -> bool:
        """Whether at least one executed step failed."""
        return any(step.failed for step in self.steps)

    @property
    def errored(self) -> bool:
        """Whether at least one executed step errored."""
        return any(step.errored for step in self.steps)

    @property
    def skipped(self) -> bool:
        """Whether all executed steps were skipped."""
        return len(self.steps) > 0 and all(step.skipped for step in self.steps)


class TestCaseResult(BaseModel):
    """Immutable summary of a test case execution with full run history.

    Attributes
    ----------
    all_runs:
        List of check results for each run. Each inner list contains the
        CheckResults from one execution of the test case.
    duration_ms:
        Total execution time in milliseconds across all runs.
    total_runs:
        Number of runs actually executed (may be less than max_runs if stopped early).
    """

    results: list[CheckResult] = Field(..., description="Check results for each run")
    duration_ms: int = Field(..., description="Total execution time in milliseconds")

    @property
    def passed(self) -> bool:
        """True when all checks passed in the final run, or when there are no checks."""
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> bool:
        """True when at least one check failed and none errored in the final run."""
        return (
            not self.errored
            and any(result.failed for result in self.results)
            and len(self.results) > 0
        )

    @property
    def errored(self) -> bool:
        """True when at least one check errored in the final run."""
        return any(result.errored for result in self.results) and len(self.results) > 0

    @property
    def skipped(self) -> bool:
        """True when all checks were skipped in the final run."""
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
