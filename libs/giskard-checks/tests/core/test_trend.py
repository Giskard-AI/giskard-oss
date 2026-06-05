"""Unit tests for SuiteRunTrendAnalyzer and related data classes."""

from datetime import datetime

import pytest
from giskard.checks import SuiteRunTrendAnalyzer
from giskard.checks.core.interaction import Trace
from giskard.checks.core.result import (
    CheckResult,
    CheckStatus,
    ScenarioResult,
    SuiteResult,
    TestCaseResult,
)
from giskard.checks.core.trend import SuiteRunPoint, SuiteRunTrendReport, SuiteTrend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_suite_result(pass_rate: float, total: int = 10) -> SuiteResult:
    """Build a minimal SuiteResult with the desired pass_rate.

    Parameters
    ----------
    pass_rate : float
        Target pass rate (fraction between 0.0 and 1.0).
    total : int
        Total number of scenarios to simulate.
    """
    passed = round(pass_rate * total)
    failed = total - passed

    trace = Trace()
    passing = [
        ScenarioResult(
            scenario_name=f"pass_{i}", steps=[], duration_ms=1, final_trace=trace
        )
        for i in range(passed)
    ]
    failing_tc = TestCaseResult(
        results=[CheckResult(status=CheckStatus.FAIL)], duration_ms=1
    )
    failing = [
        ScenarioResult(
            scenario_name=f"fail_{i}",
            steps=[failing_tc],
            duration_ms=1,
            final_trace=trace,
        )
        for i in range(failed)
    ]

    return SuiteResult(results=passing + failing, duration_ms=10)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_defaults():
    """Default window and thresholds are applied."""
    analyzer = SuiteRunTrendAnalyzer()
    assert analyzer.window == 10
    assert analyzer.regression_threshold == -0.01
    assert analyzer.improvement_threshold == 0.005


def test_init_custom_params():
    """Custom window, regression_threshold, and improvement_threshold are stored."""
    analyzer = SuiteRunTrendAnalyzer(
        window=5, regression_threshold=-0.02, improvement_threshold=0.01
    )
    assert analyzer.window == 5
    assert analyzer.regression_threshold == -0.02
    assert analyzer.improvement_threshold == 0.01


def test_init_window_less_than_two_raises():
    """window < 2 should raise ValueError."""
    with pytest.raises(ValueError, match="window must be >= 2"):
        SuiteRunTrendAnalyzer(window=1)


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


def test_record_appends_runs():
    """record() stores results internally."""
    analyzer = SuiteRunTrendAnalyzer()
    result = _make_suite_result(1.0)
    analyzer.record(result)
    assert len(analyzer._runs) == 1


def test_record_uses_provided_timestamp():
    """record() uses the given timestamp when provided."""
    analyzer = SuiteRunTrendAnalyzer()
    ts = datetime(2024, 4, 1, 12, 0, 0)
    analyzer.record(_make_suite_result(0.9), timestamp=ts)
    _, stored_ts = analyzer._runs[0]
    assert stored_ts == ts


def test_record_fills_timestamp_when_omitted():
    """record() fills a datetime when no timestamp is provided."""
    analyzer = SuiteRunTrendAnalyzer()
    analyzer.record(_make_suite_result(0.9))
    _, stored_ts = analyzer._runs[0]
    assert isinstance(stored_ts, datetime)


# ---------------------------------------------------------------------------
# analyze() — guard conditions
# ---------------------------------------------------------------------------


def test_analyze_requires_at_least_two_runs():
    """analyze() with fewer than 2 recorded runs raises ValueError."""
    analyzer = SuiteRunTrendAnalyzer()
    analyzer.record(_make_suite_result(0.9))
    with pytest.raises(ValueError, match="At least 2 runs"):
        analyzer.analyze()


def test_analyze_with_no_runs_raises():
    """analyze() with zero recorded runs raises ValueError."""
    analyzer = SuiteRunTrendAnalyzer()
    with pytest.raises(ValueError, match="At least 2 runs"):
        analyzer.analyze()


# ---------------------------------------------------------------------------
# analyze() — regression detection
# ---------------------------------------------------------------------------


def test_analyze_detects_degrading_trend():
    """A steadily declining pass_rate is classified as degrading."""
    analyzer = SuiteRunTrendAnalyzer(regression_threshold=-0.01)
    # pass_rates: 1.0, 0.9, 0.8, 0.7, 0.6 → slope ≈ -0.1/run
    for rate in [1.0, 0.9, 0.8, 0.7, 0.6]:
        analyzer.record(_make_suite_result(rate))

    report = analyzer.analyze()
    assert report.pass_rate_trend.direction == "degrading"
    assert report.pass_rate_trend.is_regression is True
    assert report.any_regression is True
    assert report.pass_rate_trend.slope < -0.01


def test_analyze_detects_improving_trend():
    """A steadily rising pass_rate is classified as improving."""
    analyzer = SuiteRunTrendAnalyzer(improvement_threshold=0.005)
    # pass_rates: 0.6, 0.7, 0.8, 0.9, 1.0 → slope ≈ +0.1/run
    for rate in [0.6, 0.7, 0.8, 0.9, 1.0]:
        analyzer.record(_make_suite_result(rate))

    report = analyzer.analyze()
    assert report.pass_rate_trend.direction == "improving"
    assert report.pass_rate_trend.is_regression is False
    assert report.any_regression is False
    assert report.pass_rate_trend.slope > 0.005


def test_analyze_detects_stable_trend():
    """A flat pass_rate is classified as stable."""
    analyzer = SuiteRunTrendAnalyzer(
        regression_threshold=-0.01, improvement_threshold=0.005
    )
    # All the same pass_rate → slope == 0.0
    for _ in range(5):
        analyzer.record(_make_suite_result(0.8))

    report = analyzer.analyze()
    assert report.pass_rate_trend.direction == "stable"
    assert report.pass_rate_trend.is_regression is False
    assert report.any_regression is False


# ---------------------------------------------------------------------------
# analyze() — window slicing
# ---------------------------------------------------------------------------


def test_analyze_respects_window():
    """analyze() only uses the most recent `window` runs."""
    analyzer = SuiteRunTrendAnalyzer(window=3)
    # First two runs are degrading; the last three are stable at 0.9
    for rate in [1.0, 0.5]:
        analyzer.record(_make_suite_result(rate))
    for _ in range(3):
        analyzer.record(_make_suite_result(0.9))

    report = analyzer.analyze()
    # Only the 3 stable runs should be analysed → window == 3
    assert report.window == 3
    assert len(report.run_points) == 3


def test_analyze_window_larger_than_recorded():
    """window > number of recorded runs uses all recorded runs."""
    analyzer = SuiteRunTrendAnalyzer(window=20)
    for rate in [0.8, 0.9]:
        analyzer.record(_make_suite_result(rate))

    report = analyzer.analyze()
    assert report.window == 2
    assert len(report.run_points) == 2


# ---------------------------------------------------------------------------
# analyze() — report structure
# ---------------------------------------------------------------------------


def test_report_run_points_count():
    """run_points length matches the number of runs in the window."""
    analyzer = SuiteRunTrendAnalyzer(window=4)
    rates = [0.8, 0.85, 0.9, 0.95]
    for rate in rates:
        analyzer.record(_make_suite_result(rate))

    report = analyzer.analyze()
    assert len(report.run_points) == 4


def test_report_run_points_indices():
    """run_index values are zero-based consecutive integers."""
    analyzer = SuiteRunTrendAnalyzer(window=3)
    for rate in [0.7, 0.8, 0.9]:
        analyzer.record(_make_suite_result(rate))

    report = analyzer.analyze()
    indices = [p.run_index for p in report.run_points]
    assert indices == [0, 1, 2]


def test_report_run_points_pass_rate():
    """run_points carry the original pass_rate values."""
    analyzer = SuiteRunTrendAnalyzer(window=3)
    rates = [0.6, 0.7, 0.8]
    for rate in rates:
        analyzer.record(_make_suite_result(rate, total=10))

    report = analyzer.analyze()
    for point, expected_rate in zip(report.run_points, rates):
        assert abs(point.pass_rate - expected_rate) < 1e-9


def test_report_run_points_counts():
    """SuiteRunPoint captures passed/failed/errored/total counts."""
    analyzer = SuiteRunTrendAnalyzer()
    # 8 passed, 2 failed out of 10
    analyzer.record(_make_suite_result(0.8, total=10))
    analyzer.record(_make_suite_result(0.8, total=10))

    report = analyzer.analyze()
    point = report.run_points[0]
    assert point.total_count == 10
    assert point.passed_count == 8
    assert point.failed_count == 2
    assert point.errored_count == 0


def test_report_run_points_timestamp():
    """Timestamps recorded via record() are preserved in run_points."""
    analyzer = SuiteRunTrendAnalyzer()
    ts1 = datetime(2024, 4, 1)
    ts2 = datetime(2024, 4, 3)
    analyzer.record(_make_suite_result(0.9), timestamp=ts1)
    analyzer.record(_make_suite_result(0.8), timestamp=ts2)

    report = analyzer.analyze()
    assert report.run_points[0].timestamp == ts1
    assert report.run_points[1].timestamp == ts2


# ---------------------------------------------------------------------------
# Public API — importable from giskard.checks
# ---------------------------------------------------------------------------


def test_public_api_importable():
    """Trend classes are importable from the top-level giskard.checks namespace."""
    from giskard.checks import (
        SuiteRunPoint,
        SuiteRunTrendAnalyzer,
        SuiteRunTrendReport,
        SuiteTrend,
    )

    assert SuiteRunTrendAnalyzer is not None
    assert SuiteRunTrendReport is not None
    assert SuiteTrend is not None
    assert SuiteRunPoint is not None


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


def test_suite_run_point_is_frozen():
    """SuiteRunPoint instances are immutable."""
    point = SuiteRunPoint(
        run_index=0,
        timestamp=None,
        pass_rate=0.9,
        passed_count=9,
        failed_count=1,
        errored_count=0,
        total_count=10,
    )
    with pytest.raises((AttributeError, TypeError)):
        point.pass_rate = 0.5  # type: ignore[misc]


def test_suite_trend_is_frozen():
    """SuiteTrend instances are immutable."""
    trend = SuiteTrend(slope=-0.05, direction="degrading", is_regression=True)
    with pytest.raises((AttributeError, TypeError)):
        trend.slope = 0.0  # type: ignore[misc]


def test_suite_run_trend_report_is_frozen():
    """SuiteRunTrendReport instances are immutable."""
    report = SuiteRunTrendReport()
    with pytest.raises((AttributeError, TypeError)):
        report.any_regression = True  # type: ignore[misc]
