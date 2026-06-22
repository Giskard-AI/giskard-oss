"""Tests for NLP metric checks."""

from typing import Any

import pytest
from giskard.checks import Check, CheckStatus, Interaction, Readability, Trace
from giskard.checks.builtin import nlp_metrics
from giskard.checks.core.extraction import NoMatch
from pydantic import ValidationError

SIMPLE_TEXT = "The cat sat on the mat. It was a sunny day."
COMPLEX_TEXT = (
    "Notwithstanding the aforementioned considerations, the implementation "
    "necessitates a comprehensive evaluation of interdependent methodological "
    "constraints before any conclusive determination can be reasonably made."
)


async def test_flesch_reading_ease_passes_for_simple_text() -> None:
    check = Readability(metric="flesch_reading_ease", min_score=60)
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain simply", outputs=SIMPLE_TEXT)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.metrics[0].name == "flesch_reading_ease"
    assert result.metrics[0].value == result.details["score"]
    assert result.details["score"] >= 60


async def test_flesch_reading_ease_fails_for_complex_text() -> None:
    check = Readability(metric="flesch_reading_ease", min_score=60)
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain simply", outputs=COMPLEX_TEXT)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "below the minimum" in result.message
    assert result.details["score"] < 60


@pytest.mark.parametrize(
    "metric",
    ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"],
)
async def test_readability_supports_metric_variants(
    metric: nlp_metrics.ReadabilityMetric,
) -> None:
    check = Readability(metric=metric)
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs=SIMPLE_TEXT)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.metrics[0].name == metric
    assert isinstance(result.metrics[0].value, float)
    assert result.details["metric"] == metric


async def test_max_score_threshold_fails_when_score_is_too_high() -> None:
    check = Readability(metric="flesch_kincaid_grade", max_score=-1)
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs=SIMPLE_TEXT)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "above the maximum" in result.message


async def test_nested_jsonpath_extraction() -> None:
    check = Readability(key="trace.last.outputs.answer", metric="gunning_fog")
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs={"answer": SIMPLE_TEXT})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["text"] == SIMPLE_TEXT


async def test_missing_key_fails() -> None:
    check = Readability(key="trace.last.outputs.missing")
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs={"answer": SIMPLE_TEXT})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert isinstance(result.details["text"], NoMatch)
    assert result.message == "No value found for key 'trace.last.outputs.missing'."


async def test_non_string_value_fails() -> None:
    check = Readability()
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs={"answer": SIMPLE_TEXT})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "is not a string" in result.message


async def test_missing_textstat_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_textstat(name: str) -> Any:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(nlp_metrics, "import_module", raise_missing_textstat)

    check = Readability()
    trace = await Trace.from_interactions(
        Interaction(inputs="Explain", outputs=SIMPLE_TEXT)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert result.message is not None
    assert "giskard-checks[nlp]" in result.message
    assert result.details["error"] == result.message


def test_threshold_validation() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Readability(min_score=10, max_score=5)

    assert "min_score must be less than or equal to max_score" in str(exc_info.value)


def test_readability_is_exported() -> None:
    assert Readability.__name__ == "Readability"


def test_readability_serialization_roundtrip() -> None:
    check = Readability(
        key="trace.last.outputs.answer",
        metric="gunning_fog",
        min_score=1,
        max_score=12,
    )

    data = check.model_dump()
    restored = Check.model_validate(data)

    assert data["kind"] == "readability"
    assert isinstance(restored, Readability)
    assert restored.key == "trace.last.outputs.answer"
    assert restored.metric == "gunning_fog"
    assert restored.min_score == 1
    assert restored.max_score == 12
