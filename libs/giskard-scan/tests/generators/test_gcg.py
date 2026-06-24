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


def test_cross_product_count_and_prompts(monkeypatch):
    base = [_base_scenario("PROMPT_A", name="A"), _base_scenario("PROMPT_B", name="B")]
    scenarios = _load(monkeypatch, base)

    # 2 base prompts x 13 suffixes, every (prompt, suffix) pair exactly once
    assert len(scenarios) == 2 * len(_SUFFIXES)
    assert {_prompt_of(s) for s in scenarios} == {
        f"{prompt}{suffix}"
        for prompt in ("PROMPT_A", "PROMPT_B")
        for suffix in _SUFFIXES
    }


def test_names_are_unique_and_gcg_prefixed(monkeypatch):
    scenarios = _load(monkeypatch, [_base_scenario("PROMPT_A", name="HarmBench #0")])
    names = [s.name for s in scenarios]

    assert len(set(names)) == len(_SUFFIXES)  # all unique
    assert all(name.startswith("GCG - HarmBench #0 [suffix #") for name in names)
    assert "GCG - HarmBench #0 [suffix #0]" in names


def test_each_base_prompt_tagged_with_every_suffix_index(monkeypatch):
    scenarios = _load(monkeypatch, [_base_scenario("PROMPT_A", name="A")])
    suffix_tags = sorted(
        tag for s in scenarios for tag in s.tags if tag.startswith("gcg-suffix:")
    )
    assert suffix_tags == sorted(f"gcg-suffix:{i}" for i in range(len(_SUFFIXES)))


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
