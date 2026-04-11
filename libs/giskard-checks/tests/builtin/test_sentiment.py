"""Tests for the Sentiment check.

These tests rely on textblob being installed in the test environment.
"""

import pytest
from giskard.checks import CheckStatus, Interaction, Sentiment, Trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(outputs: str | dict) -> Trace:  # type: ignore[type-arg]
    return Trace(interactions=[Interaction(inputs={"query": "test"}, outputs=outputs)])


# ---------------------------------------------------------------------------
# Label-based assertions
# ---------------------------------------------------------------------------


async def test_positive_text_passes_with_expected_positive() -> None:
    """Clearly positive text should pass when expected='positive'."""
    check = Sentiment(
        text="This product is absolutely fantastic and I love it!",
        expected="positive",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["sentiment_label"] == "positive"
    assert result.details["polarity_score"] > 0


async def test_negative_text_passes_with_expected_negative() -> None:
    """Clearly negative text should pass when expected='negative'."""
    check = Sentiment(
        text="This was a terrible experience. I am very disappointed.",
        expected="negative",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["sentiment_label"] == "negative"
    assert result.details["polarity_score"] < 0


async def test_neutral_text_passes_with_expected_neutral() -> None:
    """Factual / neutral text should pass when expected='neutral'."""
    check = Sentiment(
        text="The meeting is scheduled for Monday at 10:00.",
        expected="neutral",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["sentiment_label"] == "neutral"


async def test_wrong_expected_label_fails() -> None:
    """Positive text should fail when expected='negative'."""
    check = Sentiment(
        text="Excellent quality, highly recommend!",
        expected="negative",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "Expected sentiment 'negative'" in result.message
    assert "positive" in result.message


# ---------------------------------------------------------------------------
# Numeric score constraints
# ---------------------------------------------------------------------------


async def test_score_above_min_passes() -> None:
    """Positive text polarity should satisfy a min_score > 0."""
    check = Sentiment(
        text="I absolutely love this, it's wonderful!",
        min_score=0.1,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["polarity_score"] >= 0.1


async def test_score_below_min_fails() -> None:
    """Negative text polarity should fail a min_score > 0."""
    check = Sentiment(
        text="Awful, terrible, I hate it.",
        min_score=0.3,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "below the minimum threshold" in result.message


async def test_score_below_max_passes() -> None:
    """Negative text polarity should satisfy a max_score < 0."""
    check = Sentiment(
        text="Very bad and disappointing.",
        max_score=-0.1,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["polarity_score"] <= -0.1


async def test_score_above_max_fails() -> None:
    """Positive text polarity should fail a max_score < 0."""
    check = Sentiment(
        text="Brilliant, amazing, perfect!",
        max_score=-0.1,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "exceeds the maximum threshold" in result.message


async def test_combined_label_and_score_both_satisfied() -> None:
    """Check that passes when both label and score constraints are met."""
    check = Sentiment(
        text="Great service, very happy!",
        expected="positive",
        min_score=0.05,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS


async def test_combined_label_and_score_label_fails() -> None:
    """Check that fails when label constraint is violated even if score is OK."""
    check = Sentiment(
        text="Good but could be better.",
        expected="negative",  # likely positive/neutral — should fail
        min_score=-1.0,  # always satisfied
    )
    result = await check.run(Trace())

    # neutral or positive text → "negative" expectation should fail
    assert result.status in (CheckStatus.FAIL, CheckStatus.PASS)
    # If textblob scores it as "positive" or "neutral", it must fail
    if result.details["sentiment_label"] != "negative":
        assert result.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Metrics attachment
# ---------------------------------------------------------------------------


async def test_polarity_score_metric_is_attached() -> None:
    """The polarity_score Metric should always be present on the result."""
    check = Sentiment(text="Pretty good overall.")
    result = await check.run(Trace())

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.name == "polarity_score"
    assert -1.0 <= metric.value <= 1.0


async def test_polarity_score_metric_matches_details() -> None:
    """The Metric value should equal the polarity_score in details."""
    check = Sentiment(text="Fantastic and wonderful!")
    result = await check.run(Trace())

    metric_value = result.metrics[0].value
    details_value = result.details["polarity_score"]
    assert abs(metric_value - details_value) < 1e-9


# ---------------------------------------------------------------------------
# JSONPath extraction
# ---------------------------------------------------------------------------


async def test_text_extracted_from_trace_default_key() -> None:
    """Text should be extracted from trace.last.outputs when not provided directly."""
    check = Sentiment(expected="positive")
    trace = _make_trace("Absolutely wonderful, I am very pleased!")
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert "wonderful" in result.details["text"]


async def test_text_extracted_from_custom_key() -> None:
    """Text should be extracted using a custom JSONPath key."""
    check = Sentiment(
        text_key="trace.last.outputs.reply",
        expected="positive",
    )
    trace = Trace(
        interactions=[
            Interaction(
                inputs={"query": "test"},
                outputs={"reply": "Excellent, very happy with the outcome!"},
            )
        ]
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["text"] == "Excellent, very happy with the outcome!"


async def test_missing_key_returns_error() -> None:
    """A non-existent JSONPath key should yield an ERROR result."""
    check = Sentiment(text_key="trace.last.outputs.nonexistent")
    trace = _make_trace({"other_field": "hello"})
    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.message is not None
    assert "nonexistent" in result.message or "No value found" in result.message


# ---------------------------------------------------------------------------
# No constraint — just measure
# ---------------------------------------------------------------------------


async def test_no_constraints_always_passes() -> None:
    """Without expected / min_score / max_score, the check always passes."""
    for text in [
        "Absolutely amazing!",
        "Terrible disaster.",
        "The door is blue.",
    ]:
        check = Sentiment(text=text)
        result = await check.run(Trace())
        assert result.status == CheckStatus.PASS, f"Expected PASS for: {text!r}"


# ---------------------------------------------------------------------------
# Registration & serialisation
# ---------------------------------------------------------------------------


async def test_check_is_serialisable() -> None:
    """Check can be serialised to a dict with the correct 'kind' field."""
    check = Sentiment(expected="positive", min_score=0.0)
    data = check.model_dump()

    assert data["kind"] == "sentiment"
    assert data["expected"] == "positive"
    assert data["min_score"] == 0.0


async def test_check_round_trips_via_discriminated_union() -> None:
    """Check can be reconstructed from a serialised dict."""
    from giskard.checks.core.check import Check

    check = Sentiment(expected="negative", max_score=0.0)
    data = check.model_dump()
    reconstructed = Check.model_validate(data)

    assert isinstance(reconstructed, Sentiment)
    assert reconstructed.expected == "negative"
    assert reconstructed.max_score == 0.0


# ---------------------------------------------------------------------------
# Textblob missing (mocked)
# ---------------------------------------------------------------------------


async def test_textblob_not_installed_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When textblob is not importable, the check should return ERROR with a helpful message."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "textblob":
            raise ImportError("No module named 'textblob'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    check = Sentiment(text="Great day!")
    result = await check.run(Trace())

    assert result.status == CheckStatus.ERROR
    assert result.message is not None
    assert "textblob" in result.message.lower()
    assert "pip install" in result.message
