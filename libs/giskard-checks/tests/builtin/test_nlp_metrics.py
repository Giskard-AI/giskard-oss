from types import SimpleNamespace

import giskard.checks.builtin.nlp_metrics as nlp_metrics
import pytest
from giskard.checks import CheckStatus, Interaction, Readability, Trace
from giskard.checks.builtin.nlp_metrics import ReadabilityMetric
from giskard.checks.core.extraction import NoMatch
from giskard.checks.core.result import Metric


@pytest.fixture
def fake_textstat(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(
        flesch_reading_ease=lambda text: {
            "Simple text.": 75.0,
            "Complex text.": 25.0,
        }[text],
        flesch_kincaid_grade=lambda text: {
            "Simple text.": 4.0,
            "Complex text.": 14.0,
        }[text],
        gunning_fog=lambda text: {
            "Simple text.": 6.0,
            "Complex text.": 18.0,
        }[text],
    )
    monkeypatch.setitem(__import__("sys").modules, "textstat", module)


async def test_readability_passes_with_min_score(fake_textstat: None) -> None:
    check = Readability(metric="flesch_reading_ease", min_score=60)
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.metrics[0].name == "flesch_reading_ease"
    assert result.metrics[0].value == 75.0
    assert result.details["score"] == 75.0


async def test_readability_fails_when_below_min_score(fake_textstat: None) -> None:
    check = Readability(metric="flesch_reading_ease", min_score=60)
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Complex text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "below the minimum threshold" in result.message
    assert result.metrics[0].value == 25.0


async def test_readability_fails_when_above_max_score(fake_textstat: None) -> None:
    check = Readability(metric="flesch_kincaid_grade", max_score=8)
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Complex text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "exceeds the maximum threshold" in result.message
    assert result.metrics[0].value == 14.0


@pytest.mark.parametrize(
    ("metric", "expected_score"),
    [
        ("flesch_reading_ease", 75.0),
        ("flesch_kincaid_grade", 4.0),
        ("gunning_fog", 6.0),
    ],
)
async def test_readability_supports_each_metric(
    fake_textstat: None, metric: ReadabilityMetric, expected_score: float
) -> None:
    check = Readability(metric=metric)
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.metrics[0].name == metric
    assert result.metrics[0].value == expected_score


async def test_readability_without_thresholds_reports_metric(
    fake_textstat: None,
) -> None:
    check = Readability(metric="gunning_fog")
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Complex text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == 18.0
    assert result.metrics == [Metric(name="gunning_fog", value=18.0)]


async def test_readability_fails_when_key_is_missing(fake_textstat: None) -> None:
    check = Readability(key="trace.last.outputs.answer")
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message == "No value found for key 'trace.last.outputs.answer'."
    assert isinstance(result.details["text"], NoMatch)


async def test_readability_fails_for_non_string_value(fake_textstat: None) -> None:
    check = Readability()
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs={"text": "x"})])

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert "must be a string" in result.message
    assert result.details["value"] == {"text": "x"}


async def test_readability_errors_when_textstat_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "textstat", raising=False)

    def fake_import_module(name: str):
        if name == "textstat":
            raise ImportError("missing textstat")
        return __import__(name)

    monkeypatch.setattr(nlp_metrics, "import_module", fake_import_module)

    check = Readability()
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])
    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.message is not None
    assert "textstat" in result.message


def test_readability_rejects_invalid_score_range() -> None:
    with pytest.raises(
        ValueError, match="min_score must be less than or equal to max_score"
    ):
        _ = Readability(min_score=10, max_score=5)
