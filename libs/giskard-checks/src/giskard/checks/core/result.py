from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.rule import Rule

from .interaction import Trace
from .protocols import RichConsoleProtocol, RichProtocol

STATUS_MAPPING = {
    "total": {
        "color": "default",
        "title": "TOTAL",
    },
    "pass": {
        "color": "green",
        "title": "✅ PASSED",
        "symbol": ".",
    },
    "error": {
        "color": "yellow",
        "title": "⚠️ ERROR",
        "symbol": "E",
    },
    "fail": {
        "color": "red",
        "title": "❌ FAILED",
        "symbol": "F",
    },
    "skip": {
        "color": "gray",
        "title": "⚠️ SKIPPED",
        "symbol": "s",
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


class BaseResult(BaseModel, frozen=True):
    def print_report(self, console: Console | None = None) -> None:
        """Format the result as a report."""
        console = console or Console()
        console.print(self)


class CheckResult(BaseResult, frozen=True):
    """Immutable result produced by running a `Check`.

    Attributes
    ----------
    status : CheckStatus
        Outcome status of the check.
    message : str or None
        Optional short message to surface to users (e.g., success/failure reason).
    metrics : list[Metric]
        Auxiliary metrics captured by the check.
    details : dict[str, Any]
        Arbitrary structured payload with additional context (e.g., failure reasons,
        timings, and any metadata the check wishes to include).
    passed : bool
        True if ``status`` is ``PASS``.
    failed : bool
        True if ``status`` is ``FAIL``.
    errored : bool
        True if ``status`` is ``ERROR``.
    skipped : bool
        True if ``status`` is ``SKIP``.
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

        if self.status == CheckStatus.FAIL or self.status == CheckStatus.ERROR:
            details = (
                self.message
                or "[dim italic]No specific error message provided[/dim italic]"
            )
        else:
            details = ""

        yield f"[{status['color']} bold]{name}[/{status['color']} bold]\t[{status['color']}]{self.status.value.upper()}[/{status['color']}]\t{details}"


class ScenarioStatus(str, Enum):
    """Outcome categories for a scenario execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class ScenarioResult[TraceType: Trace](BaseResult, frozen=True):  # pyright: ignore[reportMissingTypeArgument]
    """Result of executing an entire scenario.

    Attributes
    ----------
    scenario_name : str
        Name of the scenario that was executed.
    steps : list[TestCaseResult]
        Ordered list of test case results produced during execution.
    duration_ms : int
        Total execution time in milliseconds.
    final_trace : TraceType
        Trace state after execution, containing all interactions that occurred.
    status : ScenarioStatus
        Aggregated outcome of the scenario derived from its steps.
    passed : bool
        True when all steps passed.
    failed : bool
        True when at least one step failed and none errored.
    errored : bool
        True when at least one step errored.
    skipped : bool
        True when all steps were skipped.
    """

    scenario_name: str = Field(..., description="Scenario name")
    steps: list["TestCaseResult"]  # TODO: rename to test_cases
    duration_ms: int = Field(..., description="Total execution time in milliseconds")
    final_trace: TraceType = Field(..., description="Final trace state after execution")

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

    @property
    def failures_and_errors(self) -> list["TestCaseResult"]:
        """Return a list of test case results that failed or errored."""
        return [step for step in self.steps if step.failed or step.errored]

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


class TestCaseResult(BaseResult, frozen=True):
    """Immutable summary of a test case execution with full run history.

    Attributes
    ----------
    results : list[CheckResult]
        Check results produced during the test case execution.
    duration_ms : int
        Total execution time in milliseconds.
    status : TestCaseStatus
        Aggregated outcome of the test case derived from its results.
    passed : bool
        True when all checks passed, or when there are no checks.
    failed : bool
        True when at least one check failed and none errored.
    errored : bool
        True when at least one check errored.
    skipped : bool
        True when all checks were skipped.
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

    @property
    def failures_and_errors(self) -> list[CheckResult]:
        """Return a list of check results that failed or errored."""
        return [result for result in self.results if result.failed or result.errored]

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


class SuiteResult(BaseResult, frozen=True):
    """Aggregate result object for the suite.

    Attributes
    ----------
    results : list[ScenarioResult]
        Scenario results produced during the suite execution.
    duration_ms : int
        Total execution time in milliseconds.
    passed_count : int
        Number of scenarios that passed.
    failed_count : int
        Number of scenarios that failed.
    errored_count : int
        Number of scenarios that errored.
    skipped_count : int
        Number of scenarios that were skipped.
    pass_rate : float
        Fraction of non-skipped scenarios that passed (1.0 when all scenarios are skipped).
    """

    results: list[ScenarioResult[Any]] = Field(
        ..., description="List of scenario results"
    )
    duration_ms: int = Field(..., description="Total execution time in milliseconds")

    @computed_field
    @property
    def passed_count(self) -> int:
        """Number of passed scenarios."""
        return sum(1 for r in self.results if r.passed)

    @computed_field
    @property
    def failed_count(self) -> int:
        """Number of failed scenarios."""
        return sum(1 for r in self.results if r.failed)

    @computed_field
    @property
    def errored_count(self) -> int:
        """Number of errored scenarios."""
        return sum(1 for r in self.results if r.errored)

    @computed_field
    @property
    def skipped_count(self) -> int:
        """Number of skipped scenarios."""
        return sum(1 for r in self.results if r.skipped)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """The pass rate of the suite (passed scenarios / (total scenarios - skipped scenarios))."""
        denominator = len(self.results) - self.skipped_count
        if denominator == 0:
            return 1.0
        return self.passed_count / denominator

    @property
    def failures_and_errors(self) -> list[ScenarioResult[Any]]:
        """Return a list of scenario results that failed or errored."""
        return [r for r in self.results if r.failed or r.errored]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule("Suite Results", style="bold blue")

        # Dots view
        yield "".join(
            f"[{STATUS_MAPPING[r.status]['color']}]{STATUS_MAPPING[r.status]['symbol']}[/{STATUS_MAPPING[r.status]['color']}]"
            for r in self.results
        )
        yield ""

        failures_and_errors = self.failures_and_errors

        if failures_and_errors:
            n_loggable_failures = 20  # TODO: make this configurable

            # Details
            yield Rule("FAILURES", characters="=", style="grey")
            for f in failures_and_errors[:n_loggable_failures]:
                yield Panel(
                    f,
                    title=f.scenario_name,
                    border_style=f"{STATUS_MAPPING[f.status]['color']} bold",
                )
            if len(failures_and_errors) > n_loggable_failures:
                yield f"  ... and {len(failures_and_errors) - n_loggable_failures} more"

            # Summary
            yield Rule("SUMMARY", characters="=", style="grey")
            for f in failures_and_errors[:n_loggable_failures]:
                status = STATUS_MAPPING[f.status]
                yield f"[{status['color']} bold]{f.scenario_name}[/{status['color']} bold]\t[{status['color']}]{f.status.value.upper()}[/{status['color']}]"
                for tc in f.failures_and_errors:
                    for c in tc.failures_and_errors:
                        yield from (
                            f"\t{line}" for line in c.__rich_console__(console, options)
                        )
            if len(failures_and_errors) > n_loggable_failures:
                yield f"  ... and {len(failures_and_errors) - n_loggable_failures} more"

        yield Rule(style="bold blue")

        # Summary metrics
        count_parts = []
        count_parts.append(
            f"[{STATUS_MAPPING['total']['color']} bold]{len(self.results)} total[/{STATUS_MAPPING['total']['color']} bold]"
        )
        if self.errored_count:
            count_parts.append(
                f"[{STATUS_MAPPING['error']['color']} bold]{self.errored_count} errored[/{STATUS_MAPPING['error']['color']} bold]"
            )
        if self.failed_count:
            count_parts.append(
                f"[{STATUS_MAPPING['fail']['color']} bold]{self.failed_count} failed[/{STATUS_MAPPING['fail']['color']} bold]"
            )
        if self.skipped_count:
            count_parts.append(
                f"[{STATUS_MAPPING['skip']['color']} bold]{self.skipped_count} skipped[/{STATUS_MAPPING['skip']['color']} bold]"
            )
        if self.passed_count:
            count_parts.append(
                f"[{STATUS_MAPPING['pass']['color']} bold]{self.passed_count} passed[/{STATUS_MAPPING['pass']['color']} bold]"
            )

        summary = ", ".join(count_parts)
        yield f"Summary: {summary} | Pass Rate: [default bold]{self.pass_rate:.1%}[/default bold] | Total Duration: {self.duration_ms}ms"
