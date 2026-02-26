from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.rule import Rule

from .interaction import Trace
from .protocols import RichConsoleProtocol, RichProtocol

STATUS_MAPPING = {
    "pass": {
        "color": "green",
        "title": "✅ PASSED",
    },
    "error": {
        "color": "yellow",
        "title": "⚠️ ERROR",
    },
    "fail": {
        "color": "red",
        "title": "❌ FAILED",
    },
    "skip": {
        "color": "gray",
        "title": "⚠️ SKIPPED",
    },
}


def _pluralize(count: int, word: str, plural: str | None = None) -> str:
    if count == 1:
        return f"1 {word}"
    if plural is None:
        plural = word + "s"
    return f"{count} {plural}"


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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        status = STATUS_MAPPING[self.status]

        name = self.details.get("check_name", "[dim italic]Unnamed check[/dim italic]")

        yield f"[{status['color']} bold]{name}[/{status['color']} bold]\t[{status['color']}]{self.status.value.upper()}[/{status['color']}]"

        if self.status == CheckStatus.FAIL or self.status == CheckStatus.ERROR:
            yield self.message or "No specific error message provided"


class ScenarioStatus(str, Enum):
    """Outcome categories for a scenario execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


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
    steps: list[TestCaseResult]  # TODO: rename to test_cases
    duration_ms: int = Field(..., description="Total execution time in milliseconds")
    final_trace: Trace[InputType, OutputType] = Field(
        ..., description="Final trace state after execution"
    )

    @computed_field
    @property
    def status(self) -> ScenarioStatus:
        """The status of the scenario."""
        if not self.steps:
            return ScenarioStatus.PASS

        # Priority-based evaluation
        if any(step.errored for step in self.steps):
            return ScenarioStatus.ERROR
        if any(step.failed for step in self.steps):
            return ScenarioStatus.FAIL
        if all(step.skipped for step in self.steps):
            return ScenarioStatus.SKIP

        return ScenarioStatus.PASS

    @property
    def passed(self) -> bool:
        """True when all steps passed."""
        return self.status == ScenarioStatus.PASS

    @property
    def failed(self) -> bool:
        """True when at least one step failed and none errored."""
        return self.status == ScenarioStatus.FAIL

    @property
    def errored(self) -> bool:
        """True when at least one step errored."""
        return self.status == ScenarioStatus.ERROR

    @property
    def skipped(self) -> bool:
        """True when all steps were skipped."""
        return self.status == ScenarioStatus.SKIP

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        status = STATUS_MAPPING[self.status]
        yield Rule(status["title"], style=f"{status['color']} bold")

        for step in self.steps:
            for result in step.results:
                yield from result.__rich_console__(console, options)

        yield Rule("Trace", style=f"{status['color']} bold")
        if isinstance(self.final_trace, RichConsoleProtocol):
            yield from self.final_trace.__rich_console__(console, options)
        elif isinstance(self.final_trace, RichProtocol):
            yield self.final_trace.__rich__()
        else:
            yield repr(self.final_trace)

        yield Rule(
            f"{_pluralize(len(self.steps), 'step')} in {self.duration_ms}ms",
            style=f"{status['color']} bold",
        )


class TestCaseStatus(str, Enum):
    """Outcome categories for a test case execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


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

    @computed_field
    @property
    def status(self) -> TestCaseStatus:
        """The status of the test case."""
        if not self.results:
            return TestCaseStatus.PASS

        # Priority-based evaluation
        if any(r.errored for r in self.results):
            return TestCaseStatus.ERROR
        if any(r.failed for r in self.results):
            return TestCaseStatus.FAIL
        if all(r.skipped for r in self.results):
            return TestCaseStatus.SKIP

        return TestCaseStatus.PASS

    @property
    def passed(self) -> bool:
        """True when all checks passed in the final run, or when there are no checks."""
        return self.status == TestCaseStatus.PASS

    @property
    def failed(self) -> bool:
        """True when at least one check failed and none errored in the final run."""
        return self.status == TestCaseStatus.FAIL

    @property
    def errored(self) -> bool:
        """True when at least one check errored in the final run."""
        return self.status == TestCaseStatus.ERROR

    @property
    def skipped(self) -> bool:
        """True when all checks were skipped in the final run."""
        return self.status == TestCaseStatus.SKIP

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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        status = STATUS_MAPPING[self.status]
        yield Rule(status["title"], style=f"{status['color']} bold")

        for result in self.results:
            yield from result.__rich_console__(console, options)

        # Build subtitle with counts
        counts = {
            "errored": sum(1 for r in self.results if r.errored),
            "failed": sum(1 for r in self.results if r.failed),
            "skipped": sum(1 for r in self.results if r.skipped),
            "passed": sum(1 for r in self.results if r.passed),
        }
        count_parts: list[str] = []
        if counts["errored"]:
            count_parts.append(
                f"[{STATUS_MAPPING['error']['color']} bold]{counts['errored']} errored[/{STATUS_MAPPING['error']['color']} bold]"
            )
        if counts["failed"]:
            count_parts.append(
                f"[{STATUS_MAPPING['fail']['color']} bold]{counts['failed']} failed[/{STATUS_MAPPING['fail']['color']} bold]"
            )
        if counts["skipped"]:
            count_parts.append(
                f"[{STATUS_MAPPING['skip']['color']} bold]{counts['skipped']} skipped[/{STATUS_MAPPING['skip']['color']} bold]"
            )
        if counts["passed"]:
            count_parts.append(
                f"[{STATUS_MAPPING['pass']['color']} bold]{counts['passed']} passed[/{STATUS_MAPPING['pass']['color']} bold]"
            )
        subtitle = ", ".join(count_parts) + f" in {self.duration_ms}ms"

        yield Rule(subtitle, style=f"{status['color']} bold")
