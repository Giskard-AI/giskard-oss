"""Tests for result classes (CheckResult, TestCaseResult, ScenarioResult, SuiteResult)."""

import pytest
from rich.console import Console

from giskard.checks import (
    CheckResult,
    CheckStatus,
    Metric,
    ScenarioResult,
    SuiteResult,
    TestCaseResult,
    Trace,
)


def _make_failing_scenario(name: str) -> ScenarioResult:
    """Create a ScenarioResult with a single failing check."""
    return ScenarioResult(
        scenario_name=name,
        steps=[
            TestCaseResult(
                results=[
                    CheckResult(
                        status=CheckStatus.FAIL,
                        message=f"Failure in {name}",
                        details={"check_name": f"check_{name}"},
                    ),
                ],
                duration_ms=10,
            ),
        ],
        duration_ms=10,
        final_trace=Trace(),
    )


class TestSuiteResultNLoggableFailures:
    """Tests for the configurable ``n_loggable_failures`` parameter."""

    def test_default_limit_is_20(self) -> None:
        """The default value of n_loggable_failures should be 20."""
        result = SuiteResult(
            results=[],
            duration_ms=0,
        )
        assert result.n_loggable_failures == 20

    def test_custom_limit_via_constructor(self) -> None:
        """Setting n_loggable_failures=5 should override the default."""
        result = SuiteResult(
            results=[],
            duration_ms=0,
            n_loggable_failures=5,
        )
        assert result.n_loggable_failures == 5

    def test_custom_limit_zero(self) -> None:
        """Setting n_loggable_failures=0 should suppress all failure details."""
        result = SuiteResult(
            results=[],
            duration_ms=0,
            n_loggable_failures=0,
        )
        assert result.n_loggable_failures == 0

    def test_custom_limit_large_value(self) -> None:
        """Setting n_loggable_failures to a large value should work."""
        result = SuiteResult(
            results=[],
            duration_ms=0,
            n_loggable_failures=100,
        )
        assert result.n_loggable_failures == 100

    def test_default_limit_in_console_output(self) -> None:
        """When defaults are used, the limit in __rich_console__ should be 20."""
        scenarios = [_make_failing_scenario(f"scenario_{i}") for i in range(25)]
        result = SuiteResult(
            results=scenarios,
            duration_ms=100,
        )

        # Render the result to a string
        console = Console(width=120)
        with console.capture() as capture:
            console.print(result)

        output = capture.get()

        # With 25 failures and default limit of 20, we should see "... and 5 more"
        assert "... and 5 more" in output

    def test_custom_limit_in_console_output(self) -> None:
        """A custom limit should restrict how many failures are displayed."""
        scenarios = [_make_failing_scenario(f"scenario_{i}") for i in range(25)]
        result = SuiteResult(
            results=scenarios,
            duration_ms=100,
            n_loggable_failures=3,
        )

        # Render the result to a string
        console = Console(width=120)
        with console.capture() as capture:
            console.print(result)

        output = capture.get()

        # With 25 failures and limit of 3, we should see "... and 22 more"
        assert "... and 22 more" in output

    def test_negative_limit_is_rejected(self) -> None:
        """Passing a negative n_loggable_failures should raise a validation error."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SuiteResult(
                results=[],
                duration_ms=0,
                n_loggable_failures=-1,
            )

    def test_limit_exceeds_failures_no_ellipsis(self) -> None:
        """When failures are fewer than the limit, no '... and N more' line."""
        scenarios = [_make_failing_scenario(f"scenario_{i}") for i in range(5)]
        result = SuiteResult(
            results=scenarios,
            duration_ms=100,
            n_loggable_failures=10,
        )

        console = Console(width=120)
        with console.capture() as capture:
            console.print(result)

        output = capture.get()

        # All 5 failures fit within the limit of 10 — no "and ... more"
        assert "and 5 more" not in output
        assert "and 10 more" not in output
