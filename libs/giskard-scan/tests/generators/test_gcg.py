from typing import Any

import pytest
from giskard.checks import Conformity, Scenario
from giskard.checks.core import Interact
from giskard.checks.generators.dataset import DatasetInputGenerator
from giskard.scan.generators.gcg import _SUFFIXES, GCGInjectionScenarioGenerator
from giskard.scan.generators.huggingface import HuggingFaceDatasetScenarioGenerator

AnyScenario = Scenario[Any, Any, Any]


def _base_scenario(
    prompt: str, *, name: str = "HarmBench #0", tags: list[str] | None = None
) -> AnyScenario:
    """A dataset-backed base scenario shaped like one HarmBench dataset row."""
    scenario: AnyScenario = Scenario(
        name=name,
        steps=[
            {
                "interacts": [Interact(inputs=DatasetInputGenerator(prompt=prompt))],
                "checks": [],
            }
        ],
    )
    return scenario.with_tags(tags) if tags else scenario


def _prompt_of(scenario: AnyScenario) -> str:
    interact = scenario.steps[0].interacts[0]
    assert isinstance(interact, Interact)
    assert isinstance(interact.inputs, DatasetInputGenerator)
    return interact.inputs.prompt


def _load(
    monkeypatch: pytest.MonkeyPatch, base: list[AnyScenario]
) -> list[AnyScenario]:
    """Run gcg's load_scenarios over fixed base scenarios, no network access.

    Patches the HuggingFace parent's load_scenarios to return the base
    scenarios so the fan-out logic is exercised in isolation.
    """
    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator,
        "load_scenarios",
        lambda self, description, languages: list(base),
    )
    return GCGInjectionScenarioGenerator().load_scenarios("desc", ["en"])


def test_defaults_point_at_harmbench():
    gen = GCGInjectionScenarioGenerator()
    assert gen.repo_id == "giskardai/harmbench-scenarios"
    assert gen.allow_commercial_use is True


def test_one_suffix_per_scenario_via_modulo(monkeypatch):
    base = [_base_scenario("PROMPT_A", name="A"), _base_scenario("PROMPT_B", name="B")]
    scenarios = _load(monkeypatch, base)

    assert len(scenarios) == len(base)
    assert _prompt_of(scenarios[0]) == f"PROMPT_A {_SUFFIXES[0]}"
    assert _prompt_of(scenarios[1]) == f"PROMPT_B {_SUFFIXES[1]}"


def test_names_are_unique_and_gcg_prefixed(monkeypatch):
    scenarios = _load(monkeypatch, [_base_scenario("PROMPT_A", name="HarmBench #0")])
    names = [s.name for s in scenarios]

    assert len(set(names)) == 1
    assert names == ["GCG - HarmBench #0 [suffix #0]"]


def test_suffix_index_rotates_with_scenario_position(monkeypatch):
    base = [
        _base_scenario(f"PROMPT_{i}", name=f"A{i}") for i in range(len(_SUFFIXES) + 2)
    ]
    scenarios = _load(monkeypatch, base)

    suffix_tags = [
        next(tag for tag in s.tags if tag.startswith("gcg-suffix:")) for s in scenarios
    ]
    assert suffix_tags == [f"gcg-suffix:{i % len(_SUFFIXES)}" for i in range(len(base))]


def test_dataset_tags_are_preserved(monkeypatch):
    base = _base_scenario(
        "PROMPT_A",
        name="A",
        tags=["dataset:harmbench", "threat-type:harmful-content-generation"],
    )
    scenarios = _load(monkeypatch, [base])

    # the suffix tag must be ADDED, not replace the dataset's own tags
    for s in scenarios:
        assert "dataset:harmbench" in s.tags
        assert "threat-type:harmful-content-generation" in s.tags
        assert any(tag.startswith("gcg-suffix:") for tag in s.tags)


def test_base_dataset_check_is_preserved(monkeypatch):
    base = _base_scenario("PROMPT_A", name="A")
    base.steps[0].checks.append(Conformity(rule="must refuse"))
    scenarios = _load(monkeypatch, [base])

    # the suffix fan-out must not drop the base scenario's own judge
    for s in scenarios:
        assert [type(c) for c in s.steps[0].checks] == [Conformity]


def test_empty_base_yields_nothing(monkeypatch):
    assert _load(monkeypatch, []) == []


def test_rejects_non_dataset_inputs(monkeypatch):
    bad = Scenario(
        name="bad",
        steps=[{"interacts": [Interact(inputs="hello")], "checks": []}],
    )
    with pytest.raises(TypeError, match="DatasetInputGenerator"):
        _load(monkeypatch, [bad])
