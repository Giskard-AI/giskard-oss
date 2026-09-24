"""Unit tests for ``_score_to_check`` in the garak adapter.

Covers the indeterminate-score handling (issue #2840): a NaN detector score
must never silently grade as PASS. The detector here is a stub — only
``hit_desc`` / ``pass_desc`` are read — so these tests need no garak install.
"""

import logging
import math
from typing import Any

import pytest
from giskard.scan.integrations.garak import _adapter
from giskard.scan.integrations.garak._adapter import _HIT_THRESHOLD, _score_to_check


class _StubDetector:
    hit_desc = "detector hit"
    pass_desc = "detector pass"


@pytest.fixture
def detector() -> Any:
    # Any: the stub only needs hit_desc/pass_desc, and the real Detector type
    # lives behind garak's optional import.
    return _StubDetector()


def test_none_score_skips(detector: Any) -> None:
    result = _score_to_check("fake.Detector", None, detector)
    assert result.skipped
    assert result.message == "detector returned no score"


def test_nan_score_skips_instead_of_passing(detector: Any) -> None:
    # Regression test for #2840: NaN fails every comparison, so without the
    # invalid-score guard it fell through to a PASS.
    result = _score_to_check("fake.Detector", math.nan, detector)
    assert result.skipped
    assert not result.passed
    assert "invalid" in (result.message or "")


@pytest.mark.parametrize("score", [math.inf, -math.inf, -1.0, 2.0])
def test_invalid_scores_skip(score: float, detector: Any) -> None:
    # Same indeterminate class as NaN: out-of-contract scores carry no valid
    # result. (-inf previously graded PASS, +inf graded FAIL, -1.0 graded
    # PASS, 2.0 graded FAIL — all silently.)
    result = _score_to_check("fake.Detector", score, detector)
    assert result.skipped
    assert not result.passed
    assert not result.failed


def test_invalid_score_logs_warning(
    detector: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_adapter.logger.name):
        _score_to_check("fake.Detector", math.nan, detector)
    assert any(
        "fake.Detector" in record.getMessage() and "invalid" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "pass"),
        (0.25, "pass"),
        # Strict ``>``: an exactly-at-threshold score is a pass, matching
        # garak's "uncertain, not a confirmed hit" reading of the boundary.
        (_HIT_THRESHOLD, "pass"),
        (0.500001, "fail"),
        (1.0, "fail"),
    ],
)
def test_threshold_boundaries(score: float, expected: str, detector: Any) -> None:
    result = _score_to_check("fake.Detector", score, detector)
    if expected == "pass":
        assert result.passed
        assert result.message == detector.pass_desc
    else:
        assert result.failed
        assert result.message == detector.hit_desc


def test_graded_results_carry_score_metric(detector: Any) -> None:
    for score in (0.2, 0.9):
        result = _score_to_check("fake.Detector", score, detector)
        assert [metric.value for metric in result.metrics] == [score]
        assert [metric.name for metric in result.metrics] == ["fake.Detector"]
