"""Tests for Hub format export."""

from giskard.checks.core.result import SuiteResult
from giskard.checks.export.hub import to_hub_format


def test_to_hub_format_pass_rate_null_when_empty() -> None:
    """Empty suites serialize pass_rate as JSON null for the Hub wire payload.

    Hub Metric.success_rate is already float | None; SuiteResult.pass_rate follows
    the same zero-denominator convention and ships through model_dump.
    """
    payload = to_hub_format(SuiteResult(results=[], duration_ms=0))
    assert "pass_rate" in payload
    assert payload["pass_rate"] is None
