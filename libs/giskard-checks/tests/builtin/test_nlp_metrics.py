import importlib
from types import SimpleNamespace

import pytest
from giskard.checks import Check, CheckStatus, Interaction, Readability, Trace
from giskard.checks.builtin.nlp_metrics import ReadabilityMetric
from giskard.checks.core.extraction import NoMatch
from giskard.checks.core.result import Metric
from giskard.checks.utils import optional_deps
from pydantic import ValidationError


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
        automated_readability_index=lambda text: {
            "Simple text.": 5.0,
            "Complex text.": 16.0,
        }[text],
        coleman_liau_index=lambda text: {
            "Simple text.": 7.0,
            "Complex text.": 15.0,
        }[text],
        dale_chall_readability_score=lambda text: {
            "Simple text.": 5.5,
            "Complex text.": 9.5,
        }[text],
    )
    monkeypatch.setitem(__import__("sys").modules, "textstat", module)


@pytest.fixture
def block_textstat_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, /, *args, **kwargs):
        if name == "textstat":
            raise ImportError("missing textstat")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(optional_deps.importlib, "import_module", fake_import_module)


async def test_readability_passes_with_min_score(fake_textstat: None) -> None:
    check = Readability(metric="flesch_reading_ease", min_score=60)
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.metrics[0].name == "flesch_reading_ease"
    assert result.metrics[0].value == 75.0
    assert result.details["score"] == 75.0
    assert "plain English" in result.details["score_guide"]


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
        ("automated_readability_index", 5.0),
        ("coleman_liau_index", 7.0),
        ("dale_chall_readability_score", 5.5),
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
    assert result.details["score_guide"]


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
    assert result.details["value"] == "{'text': 'x'}"


async def test_readability_returns_error_when_metric_computation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        flesch_reading_ease=lambda text: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setitem(__import__("sys").modules, "textstat", module)

    check = Readability(metric="flesch_reading_ease")
    trace = Trace(interactions=[Interaction(inputs="Prompt", outputs="Simple text.")])
    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.message is not None
    assert "Failed to compute readability score" in result.message
    assert result.details["error"] == "boom"


def test_readability_rejects_invalid_score_range(fake_textstat: None) -> None:
    with pytest.raises(
        ValueError, match="min_score must be less than or equal to max_score"
    ):
        _ = Readability(min_score=10, max_score=5)


def test_readability_requires_textstat_on_instantiation(
    block_textstat_import: None,
) -> None:
    with pytest.raises(ValidationError, match="textstat"):
        _ = Readability()


def test_readability_requires_textstat_on_model_validate_json(
    block_textstat_import: None,
) -> None:
    with pytest.raises(ValidationError, match="textstat"):
        Check.model_validate_json('{"kind":"readability"}')
