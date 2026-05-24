"""Tests for the Sentiment built-in check."""

import sys

import pytest
from giskard.checks import CheckStatus, Interaction, Sentiment, Trace
from giskard.checks.core import OptionalDependencyError

POSITIVE = "I absolutely love this product, it is fantastic!"
NEGATIVE = "Terrible experience, completely disappointed."
NEUTRAL = "The package arrived on Tuesday."


async def test_label_match_passes() -> None:
    """Expected label matches the analysed polarity."""
    result = await Sentiment(text=POSITIVE, expected="positive").run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["sentiment_label"] == "positive"


async def test_label_mismatch_fails() -> None:
    """Expected label differs from the analysed polarity."""
    result = await Sentiment(text=NEGATIVE, expected="positive").run(Trace())
    assert result.status == CheckStatus.FAIL
    assert "expected 'positive'" in (result.message or "")


async def test_score_within_range_passes() -> None:
    """Polarity inside the configured score window."""
    result = await Sentiment(text=POSITIVE, min_score=0.3).run(Trace())
    assert result.status == CheckStatus.PASS
    [metric] = result.metrics
    assert metric.name == "sentiment_polarity"
    assert metric.value >= 0.3


async def test_score_out_of_range_fails() -> None:
    """Polarity outside the configured score window."""
    result = await Sentiment(text=NEGATIVE, min_score=0.0).run(Trace())
    assert result.status == CheckStatus.FAIL
    assert "outside the required range" in (result.message or "")


async def test_extracts_text_from_trace_default_key() -> None:
    """When `text` is not given, the default `text_key` is used."""
    interaction = Interaction(inputs="How was it?", outputs=POSITIVE)
    result = await Sentiment(expected="positive").run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS


async def test_missing_text_returns_failure() -> None:
    """No text in trace and no direct `text` → failure with a clear message."""
    result = await Sentiment(expected="positive").run(Trace())
    assert result.status == CheckStatus.FAIL
    assert "No value found for text key" in (result.message or "")


async def test_missing_textblob_raises_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``textblob`` is unavailable, the check raises OptionalDependencyError."""
    monkeypatch.setitem(sys.modules, "textblob", None)
    with pytest.raises(OptionalDependencyError, match=r"giskard-checks\[nlp\]"):
        await Sentiment(text=POSITIVE).run(Trace())
