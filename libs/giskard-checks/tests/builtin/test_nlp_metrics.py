"""Tests for the Readability check."""

from unittest.mock import patch

import pytest

from giskard.checks import CheckStatus, Interaction, Trace
from giskard.checks.builtin.nlp_metrics import Readability
from giskard.checks.core.result import Metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(text: str) -> Trace:
      return Trace(interactions=[Interaction(inputs="prompt", outputs=text)])


SIMPLE_TEXT = "The cat sat on the mat. Dogs run fast."
COMPLEX_TEXT = (
      "The epistemological ramifications of postmodernist deconstruction "
      "necessitate a comprehensive reassessment of contemporary hermeneutical frameworks."
)


# ---------------------------------------------------------------------------
# Basic pass / fail
# ---------------------------------------------------------------------------


async def test_simple_text_passes_flesch_reading_ease_min_score() -> None:
      """Test that simple text passes a low Flesch Reading Ease threshold."""
      check = Readability(metric="flesch_reading_ease", min_score=50)
      result = await check.run(_make_trace(SIMPLE_TEXT))
      assert result.status == CheckStatus.PASS


async def test_complex_text_fails_flesch_reading_ease_min_score() -> None:
      """Test that complex text fails a high Flesch Reading Ease threshold."""
      check = Readability(metric="flesch_reading_ease", min_score=80)
      result = await check.run(_make_trace(COMPLEX_TEXT))
      assert result.status == CheckStatus.FAIL
      assert result.message is not None
      assert "below the minimum threshold" in result.message


async def test_simple_text_passes_flesch_kincaid_grade_max_score() -> None:
      """Test that simple text passes a low Flesch-Kincaid Grade max threshold."""
      check = Readability(metric="flesch_kincaid_grade", max_score=5)
      result = await check.run(_make_trace(SIMPLE_TEXT))
      assert result.status == CheckStatus.PASS


async def test_complex_text_fails_flesch_kincaid_grade_max_score() -> None:
      """Test that complex text fails a low Flesch-Kincaid Grade max threshold."""
      check = Readability(metric="flesch_kincaid_grade", max_score=5)
      result = await check.run(_make_trace(COMPLEX_TEXT))
      assert result.status == CheckStatus.FAIL
      assert result.message is not None
      assert "exceeds the maximum threshold" in result.message


async def test_simple_text_passes_gunning_fog_max_score() -> None:
      """Test that simple text passes a low Gunning Fog max threshold."""
      check = Readability(metric="gunning_fog", max_score=8)
      result = await check.run(_make_trace(SIMPLE_TEXT))
      assert result.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# No thresholds - always passes when text is found
# ---------------------------------------------------------------------------


async def test_no_thresholds_always_passes() -> None:
      """Test that omitting both min_score and max_score always results in PASS."""
      check = Readability()
      result = await check.run(_make_trace(SIMPLE_TEXT))
      assert result.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# Metric is included in result
# ---------------------------------------------------------------------------


async def test_score_included_as_metric() -> None:
      """Test that the readability score is attached as a Metric on the result."""
      check = Readability(metric="flesch_reading_ease")
      result = await check.run(_make_trace(SIMPLE_TEXT))
      assert len(result.metrics) == 1
      assert isinstance(result.metrics[0], Metric)
      assert result.metrics[0].name == "flesch_reading_ease"
      assert isinstance(result.metrics[0].value, float)


async def test_metric_name_matches_selected_metric() -> None:
      """Test that the metric name in results matches the configured metric."""
      for metric in ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"):
                check = Readability(metric=metric)  # type: ignore[arg-type]
        result = await check.run(_make_trace(SIMPLE_TEXT))
        assert result.metrics[0].name == metric


# ---------------------------------------------------------------------------
# NoMatch - missing key
# ---------------------------------------------------------------------------


async def test_missing_key_returns_failure() -> None:
      """Test that a missing JSONPath key results in FAIL with a clear message."""
      check = Readability(key="trace.last.outputs.nonexistent")
      result = await check.run(Trace())
      assert result.status == CheckStatus.FAIL
      assert "No value found" in (result.message or "")


# ---------------------------------------------------------------------------
# textstat not installed
# ---------------------------------------------------------------------------


async def test_missing_textstat_returns_error() -> None:
      """Test that a missing textstat package returns CheckStatus.ERROR."""
      with patch.dict("sys.modules", {"textstat": None}):
                check = Readability()
                result = await check.run(_make_trace(SIMPLE_TEXT))
            assert result.status == CheckStatus.ERROR
    assert "textstat" in (result.message or "")


# ---------------------------------------------------------------------------
# Details dict
# ---------------------------------------------------------------------------


async def test_details_contain_expected_keys() -> None:
      """Test that the result details dict contains all expected keys."""
    check = Readability(metric="flesch_reading_ease", min_score=30)
    result = await check.run(_make_trace(SIMPLE_TEXT))
    for key in ("text", "metric", "score", "min_score", "max_score"):
              assert key in result.details


async def test_details_reflect_configured_thresholds() -> None:
      """Test that the details dict accurately reflects the configured thresholds."""
    check = Readability(metric="gunning_fog", min_score=1.0, max_score=20.0)
    result = await check.run(_make_trace(SIMPLE_TEXT))
    assert result.details["metric"] == "gunning_fog"
    assert result.details["min_score"] == 1.0
    assert result.details["max_score"] == 20.0
