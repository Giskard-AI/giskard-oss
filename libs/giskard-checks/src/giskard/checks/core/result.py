import os
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from giskard.checks.scenarios.suite import Suite

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

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

MAX_REPORTED_FAILURES_ENV_VAR = "GISKARD_CHECKS_MAX_REPORTED_FAILURES"
STATUS_SUMMARY_ORDER: tuple[tuple[str, str], ...] = (
    ("error", "errored"),
    ("fail", "failed"),
    ("skip", "skipped"),
    ("pass", "passed"),
)


def format_status_count_parts(counts: Mapping[str, int]) -> list[str]:
    """Build Rich markup fragments for non-zero status counts in summary order."""
    return [
        f"[{STATUS_MAPPING[key]['color']} bold]{counts[key]} {label}"
        f"[/{STATUS_MAPPING[key]['color']} bold]"
        for key, label in STATUS_SUMMARY_ORDER
        if counts.get(key)
    ]


def format_status_count_text(
    counts: Mapping[str, int], *, prefix: str = ""
) -> Text | None:
    """Colored counts line, or ``None`` when every count is zero."""
    parts = format_status_count_parts(counts)
    if not parts:
        return None
    return Text.from_markup(prefix + ", ".join(parts))


def _pluralize(count: int, word: str, plural: str | None = None) -> str:
    if count == 1:
        return f"1 {word}"
    if plural is None:
        plural = word + "s"
    return f"{count} {plural}"


def _max_reported_failures_from_env() -> int | None:
    """Return failure cap from env, or ``None`` for unlimited reporting."""
    raw_value = os.getenv(MAX_REPORTED_FAILURES_ENV_VAR)
    if raw_value is None:
        return None

    try:
        value = int(raw_value)
    except ValueError:
        return None

    if value < 0:
        return None

    return value


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
    multiple_runs : int
        Configured upper bound on full scenario executions for this invocation.
    runs_executed : int
        How many full scenario executions ran before stopping (≤ ``multiple_runs``).
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
    multiple_runs: int = Field(
        default=1,
        description="Configured maximum full scenario executions for this invocation.",
    )
    runs_executed: int = Field(
        default=1,
        description="Full scenario executions that ran before stopping (at most multiple_runs).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Snapshot of scenario tags at run time.",
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
            f"{_pluralize(len(self.steps), 'step')} in {self.duration_ms}ms"
            f" | runs: {self.runs_executed}/{self.multiple_runs}",
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
    last_interaction_index : int | None
        0-based index, in the scenario's final trace, of the last interaction
        this step added before its checks ran. ``None`` when the step added no
        interactions (e.g. skipped). Consumers (such as the Giskard Hub upload
        flow) use this to attribute check results to a specific interaction.
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
    last_interaction_index: int | None = Field(
        default=None,
        description=(
            "0-based index of the last trace interaction added by this step's "
            "interacts before checks ran; None when no interactions were added."
        ),
    )

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

        status_counts = {
            "error": sum(1 for r in self.results if r.errored),
            "fail": sum(1 for r in self.results if r.failed),
            "skip": sum(1 for r in self.results if r.skipped),
            "pass": sum(1 for r in self.results if r.passed),
        }
        subtitle = ", ".join(format_status_count_parts(status_counts)) + (
            f" in {self.duration_ms}ms"
        )

        yield Rule(subtitle, style=f"{status['color']} bold")


class SuiteResult(BaseResult, frozen=True):
    """Aggregate result object for the suite.

    Attributes
    ----------
    results : list[ScenarioResult]
        Scenario results produced during the suite execution.
    duration_ms : int
        Total execution time in milliseconds.
    suite : Suite | None
        The Suite that produced this result. Excluded from serialization
        (``None`` after a serialize/deserialize round-trip).
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
    suite: "Suite[Any, Any] | None" = Field(
        default=None, exclude=True, description="The Suite that produced this result"
    )

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

    def to_junit_xml(self, path: str | Path | None = None) -> str:
        from ..export.junit import to_junit_xml

        return to_junit_xml(self, path=path)

    def to_hub_format(self) -> dict[str, Any]:
        """Convert the suite result into a JSON-serializable Giskard Hub payload.

        The returned dict can be passed directly to
        :meth:`giskard_hub.HubClient.evaluations.upload` to upload the suite
        result to the Hub.
        """
        from ..export.hub import to_hub_format

        return to_hub_format(self)

    def group_by(self, key: str) -> "GroupedSuiteResult":
        """Group results by a tag key and return a GroupedSuiteResult.

        A scenario may appear in multiple buckets if it carries several tags with
        the same key (e.g. ``["Category:Hallucination", "Category:Adversarial"]``).
        This is intentional: a scenario that tests two vulnerabilities affects both
        pass rates. Totals across buckets may therefore exceed the number of scenarios.
        Scenarios with no tag matching ``key`` go into the ``None`` bucket.

        Parameters
        ----------
        key : str
            The tag key to group by (e.g. ``"Category"``).

        Returns
        -------
        GroupedSuiteResult
            Wrapper holding this result, the key, and per-group stats.
        """

        buckets: defaultdict[str | None, dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0, "errored": 0, "skipped": 0}
        )

        for result in self.results:
            matched_values: set[str] = set()
            for tag in result.tags:
                tag_key, tag_value = _parse_tag(tag)
                if tag_key == key:
                    matched_values.add(tag_value)
            for val in matched_values or {None}:
                _record_into(buckets[val], result)

        groups = {
            bucket_key: GroupStats(name=bucket_key, **counts)
            for bucket_key, counts in buckets.items()
        }

        return GroupedSuiteResult(suite_result=self, key=key, groups=groups)

    def print_report(
        self, console: Console | None = None, group_by: str | None = None
    ) -> None:
        """Print the suite report, optionally with a grouped pass-rate table.

        Parameters
        ----------
        console : Console | None
            Rich console to use. Defaults to a new Console().
        group_by : str | None
            Tag key to group by (e.g. ``"Category"``). When set, appends a
            per-group pass-rate table after the standard report.
        """
        console = console or Console()
        console.print(self.group_by(group_by) if group_by is not None else self)

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
            max_reported_failures = _max_reported_failures_from_env()
            reported_failures = failures_and_errors[:max_reported_failures]
            n_hidden = len(failures_and_errors) - len(reported_failures)

            # Details
            yield Rule("FAILURES", characters="=", style="grey")
            for f in reported_failures:
                yield Panel(
                    f,
                    title=f.scenario_name,
                    border_style=f"{STATUS_MAPPING[f.status]['color']} bold",
                )
            if n_hidden > 0:
                yield f"  ... and {n_hidden} more"

            # Summary
            yield Rule("SUMMARY", characters="=", style="grey")
            for f in reported_failures:
                status = STATUS_MAPPING[f.status]
                yield f"[{status['color']} bold]{f.scenario_name}[/{status['color']} bold]\t[{status['color']}]{f.status.value.upper()}[/{status['color']}]"
                for tc in f.failures_and_errors:
                    for c in tc.failures_and_errors:
                        yield from (
                            f"\t{line}" for line in c.__rich_console__(console, options)
                        )
            if n_hidden > 0:
                yield f"  ... and {n_hidden} more"

        yield Rule(style="bold blue")

        # Summary metrics
        count_parts = [
            f"[{STATUS_MAPPING['total']['color']} bold]{len(self.results)} total[/{STATUS_MAPPING['total']['color']} bold]"
        ]
        count_parts.extend(
            format_status_count_parts(
                {
                    "error": self.errored_count,
                    "fail": self.failed_count,
                    "skip": self.skipped_count,
                    "pass": self.passed_count,
                }
            )
        )
        summary = ", ".join(count_parts)
        yield f"Summary: {summary} | Pass Rate: [default bold]{self.pass_rate:.1%}[/default bold] | Total Duration: {self.duration_ms}ms"


def _parse_tag(tag: str) -> tuple[str, str]:
    key, _, value = tag.partition(":")
    return key, value


def _record_into(bucket: dict[str, int], result: "ScenarioResult[Any]") -> None:
    if result.passed:
        bucket["passed"] += 1
    elif result.errored:
        bucket["errored"] += 1
    elif result.skipped:
        bucket["skipped"] += 1
    else:
        bucket["failed"] += 1


class GroupStats(BaseModel, frozen=True):
    """Pass/fail counts for one tag-value bucket. Mirrors Hub's Metric shape."""

    name: str | None
    passed: int
    failed: int
    errored: int
    skipped: int = 0

    @computed_field
    @property
    def total(self) -> int:
        """Total scenarios in this bucket (passed + failed + errored + skipped)."""
        return self.passed + self.failed + self.errored + self.skipped

    @computed_field
    @property
    def non_skipped(self) -> int:
        """Scenarios counted toward pass rate (total minus skipped)."""
        return self.total - self.skipped

    @computed_field
    @property
    def pass_rate(self) -> float | None:
        """Fraction passed out of non-skipped scenarios; None when non_skipped == 0."""
        if self.non_skipped == 0:
            return None
        return self.passed / self.non_skipped


class GroupedSuiteResult(BaseResult, frozen=True):
    """SuiteResult grouped by a tag key, with per-group stats and Rich table."""

    suite_result: SuiteResult
    key: str
    groups: dict[str | None, GroupStats]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield from self.suite_result.__rich_console__(console, options)

        table = Table(title=f"Results by {self.key}")
        table.add_column(self.key, style="bold")
        table.add_column("Pass Rate", justify="right")

        for group_value, stats in self.groups.items():
            if group_value is None:
                display_name = "(untagged)"
            elif group_value == "":
                display_name = "true"
            else:
                display_name = group_value
            rate = (
                f"{stats.passed} / {stats.non_skipped}"
                if stats.pass_rate is not None
                else "—"
            )
            table.add_row(display_name, rate)

        yield table
