"""Suite run trend analysis — detect pass_rate regression across sequential runs.

This module provides :class:`SuiteRunTrendAnalyzer`, which records sequential
:class:`~giskard.checks.core.result.SuiteResult` executions and computes an
OLS (ordinary least-squares) slope over a rolling window to detect whether
``pass_rate`` is improving, stable, or degrading.

Zero external dependencies — uses :mod:`statistics` (stdlib, Python ≥ 3.10).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .result import SuiteResult

__all__ = [
    "SuiteRunPoint",
    "SuiteTrend",
    "SuiteRunTrendReport",
    "SuiteRunTrendAnalyzer",
]


@dataclass(frozen=True)
class SuiteRunPoint:
    """A single suite run's pass rate at a point in time.

    Attributes
    ----------
    run_index : int
        Zero-based ordinal position of this run within the analysis window.
    timestamp : datetime or None
        Wall-clock time when :meth:`~SuiteRunTrendAnalyzer.record` was called,
        or ``None`` if no timestamp was provided.
    pass_rate : float
        Fraction of non-skipped scenarios that passed (mirrors
        :attr:`~giskard.checks.core.result.SuiteResult.pass_rate`).
    passed_count : int
        Number of scenarios that passed.
    failed_count : int
        Number of scenarios that failed.
    errored_count : int
        Number of scenarios that errored.
    total_count : int
        Total number of scenarios (including skipped).
    """

    run_index: int
    timestamp: datetime | None
    pass_rate: float
    passed_count: int
    failed_count: int
    errored_count: int
    total_count: int


@dataclass(frozen=True)
class SuiteTrend:
    """OLS slope and direction for ``pass_rate`` across a window of runs.

    Attributes
    ----------
    slope : float
        Change in ``pass_rate`` per run index (OLS estimate).  A value of
        ``-0.03`` means the pass rate falls by 3 percentage points per run.
    direction : {"improving", "degrading", "stable"}
        Human-readable classification derived from the slope.
    is_regression : bool
        ``True`` when *direction* is ``"degrading"`` (slope is more negative
        than the configured *regression_threshold*).
    """

    slope: float
    direction: Literal["improving", "degrading", "stable"]
    is_regression: bool


@dataclass(frozen=True)
class SuiteRunTrendReport:
    """Cross-run trend analysis over a window of :class:`SuiteResult` executions.

    Attributes
    ----------
    run_points : list[SuiteRunPoint]
        Ordered snapshots for every run included in the analysis window.
    pass_rate_trend : SuiteTrend
        OLS trend computed over :attr:`run_points`.
    any_regression : bool
        ``True`` when :attr:`pass_rate_trend` flags a regression.
    window : int
        Number of run points actually used (≤ the configured window size).
    """

    run_points: list[SuiteRunPoint] = field(default_factory=list)
    pass_rate_trend: SuiteTrend = field(
        default_factory=lambda: SuiteTrend(
            slope=0.0, direction="stable", is_regression=False
        )
    )
    any_regression: bool = False
    window: int = 0


class SuiteRunTrendAnalyzer:
    """Detect pass_rate regression across sequential :class:`SuiteResult` runs.

    Records :class:`~giskard.checks.core.result.SuiteResult` objects one at a
    time and computes an OLS slope over the most recent *window* runs to
    determine whether the suite's pass rate is improving, stable, or degrading.

    Parameters
    ----------
    window : int
        Number of most recent runs to include in each analysis (default: 10).
        Must be ≥ 2 (OLS requires at least two points).
    regression_threshold : float
        Slope value below which a trend is classified as ``"degrading"`` and
        :attr:`~SuiteRunTrendReport.any_regression` is set to ``True``
        (default: ``-0.01``, i.e. a drop of 1 percentage point per run).
    improvement_threshold : float
        Slope value above which a trend is classified as ``"improving"``
        (default: ``0.005``).

    Examples
    --------
    >>> from giskard.checks.core.trend import SuiteRunTrendAnalyzer
    >>> analyzer = SuiteRunTrendAnalyzer(window=5, regression_threshold=-0.01)
    >>> analyzer.window
    5
    >>> analyzer.regression_threshold
    -0.01
    """

    def __init__(
        self,
        window: int = 10,
        regression_threshold: float = -0.01,
        improvement_threshold: float = 0.005,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 (OLS requires at least two points)")
        self.window = window
        self.regression_threshold = regression_threshold
        self.improvement_threshold = improvement_threshold
        self._runs: list[tuple[SuiteResult, datetime]] = []

    def record(
        self,
        result: SuiteResult,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a completed :class:`~giskard.checks.core.result.SuiteResult`.

        Parameters
        ----------
        result : SuiteResult
            The suite result to append to the internal run history.
        timestamp : datetime or None
            Optional wall-clock time for this run.  When omitted,
            ``datetime.now(UTC)`` is used.
        """
        self._runs.append((result, timestamp or datetime.now(UTC)))

    def analyze(self) -> SuiteRunTrendReport:
        """Compute OLS trend over the most recent *window* runs.

        Returns
        -------
        SuiteRunTrendReport
            Report containing per-run snapshots, the OLS trend, and a
            regression flag.

        Raises
        ------
        ValueError
            If fewer than two runs have been recorded (OLS requires at least
            two data points).

        Examples
        --------
        >>> # Doctest omitted — requires SuiteResult fixtures.
        ... pass
        """
        if len(self._runs) < 2:
            raise ValueError(
                "At least 2 runs must be recorded before calling analyze()"
            )

        recent = self._runs[-self.window :]

        points: list[SuiteRunPoint] = [
            SuiteRunPoint(
                run_index=i,
                timestamp=ts,
                pass_rate=r.pass_rate,
                passed_count=r.passed_count,
                failed_count=r.failed_count,
                errored_count=r.errored_count,
                total_count=len(r.results),
            )
            for i, (r, ts) in enumerate(recent)
        ]

        xs = [float(p.run_index) for p in points]
        ys = [p.pass_rate for p in points]
        regression = statistics.linear_regression(xs, ys)
        slope: float = regression.slope

        direction: Literal["improving", "degrading", "stable"]
        if slope > self.improvement_threshold:
            direction = "improving"
        elif slope < self.regression_threshold:
            direction = "degrading"
        else:
            direction = "stable"

        pass_rate_trend = SuiteTrend(
            slope=slope,
            direction=direction,
            is_regression=(direction == "degrading"),
        )

        return SuiteRunTrendReport(
            run_points=points,
            pass_rate_trend=pass_rate_trend,
            any_regression=pass_rate_trend.is_regression,
            window=len(points),
        )
