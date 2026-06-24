"""Tests that MISSING fields omit from JSON and round-trip as MISSING."""

import json

import pytest
from giskard.checks.builtin.comparison import Equals
from giskard.checks.builtin.text_matching import RegexMatching, StringMatching
from giskard.checks.core.interaction.interact import Interact
from giskard.checks.core.scenario import Scenario
from giskard.checks.judges.groundedness import Groundedness
from giskard.checks.judges.toxicity import Toxicity
from giskard.checks.scenarios.runner import _build_steps
from giskard.checks.scenarios.suite import Suite
from pydantic.experimental.missing_sentinel import MISSING


def _assert_missing_json_roundtrip(model, field: str) -> None:
    """Assert a MISSING field is absent from JSON and restored after round-trip."""
    assert getattr(model, field) is MISSING

    payload = json.loads(model.model_dump_json())
    assert field not in payload, (
        f"{field!r} must be omitted from JSON, got {payload.get(field)!r}"
    )

    restored = type(model).model_validate_json(model.model_dump_json())
    assert getattr(restored, field) is MISSING


@pytest.mark.parametrize(
    ("model", "field"),
    [
        pytest.param(Interact(inputs="hi"), "outputs", id="interact-outputs"),
        pytest.param(
            Equals(
                key="trace.last.outputs",
                expected_value_key="trace.last.metadata.expected",
            ),
            "expected_value",
            id="equals-expected-value",
        ),
        pytest.param(
            StringMatching(text="hello", keyword_key="trace.last.inputs.keyword"),
            "keyword",
            id="string-matching-keyword",
        ),
        pytest.param(
            RegexMatching(text="hello", pattern_key="trace.last.inputs.pattern"),
            "pattern",
            id="regex-matching-pattern",
        ),
        pytest.param(Scenario("scenario"), "target", id="scenario-target"),
        pytest.param(Suite(name="suite"), "target", id="suite-target"),
        pytest.param(Toxicity(), "output", id="toxicity-output"),
        pytest.param(Groundedness(), "answer", id="groundedness-answer"),
    ],
)
def test_missing_field_json_roundtrip(model, field):
    _assert_missing_json_roundtrip(model, field)


def test_scenario_nested_interact_outputs_missing_json_roundtrip():
    scenario = Scenario("nested").interact("hello")
    interact = scenario.steps[0].interacts[0]
    assert isinstance(interact, Interact)
    assert interact.outputs is MISSING

    payload = json.loads(scenario.model_dump_json())
    nested_outputs = payload["steps"][0]["interacts"][0]
    assert "outputs" not in nested_outputs
    assert nested_outputs.get("outputs") is None

    restored = Scenario.model_validate_json(scenario.model_dump_json())
    restored_interact = restored.steps[0].interacts[0]
    assert isinstance(restored_interact, Interact)
    assert restored_interact.outputs is MISSING


def test_missing_outputs_still_binds_target_after_json_roundtrip():
    async def target(inputs: str, trace) -> str:
        return f"echo: {inputs}"

    scenario = Scenario("bind-target").interact("hello")
    restored = Scenario.model_validate_json(scenario.model_dump_json())

    steps = _build_steps(restored, target)
    interact = steps[0].interacts[0]
    assert isinstance(interact, Interact)
    assert interact.outputs is target
