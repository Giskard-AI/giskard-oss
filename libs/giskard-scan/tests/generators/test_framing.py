from typing import Any

import pytest
from giskard.checks import (
    DatasetInputGenerator,
    Interact,
    LLMGenerator,
    LLMJudge,
    Scenario,
)
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.framing import (
    CitationFramingScenarioGenerator,
    FramingScenarioGenerator,
    GrandmaFramingScenarioGenerator,
    LikertFramingScenarioGenerator,
    MathFramingScenarioGenerator,
)
from giskard.scan.generators.huggingface import HuggingFaceDatasetScenarioGenerator

AnyScenario = Scenario[Any, Any, Any]

_VARIANTS = [
    (GrandmaFramingScenarioGenerator, "Grandma Framing", "grandma_framing.j2", 1),
    (MathFramingScenarioGenerator, "Math Framing", "math_framing.j2", 1),
    (CitationFramingScenarioGenerator, "Citation Framing", "citation_framing.j2", 1),
    (LikertFramingScenarioGenerator, "Likert Framing", "likert_framing.j2", 2),
]


def _base_scenario(
    behavior: str,
    *,
    name: str = "HarmBench #0",
    functional_category: str = "standard",
    tags: list[str] | None = None,
) -> AnyScenario:
    """A dataset-backed base scenario shaped like one HarmBench dataset row."""
    scenario: AnyScenario = Scenario(
        name=name,
        steps=[
            {
                "interacts": [Interact(inputs=DatasetInputGenerator(prompt=behavior))],
                "checks": [
                    LLMJudge(prompt_path="giskard.scan::judges/harmbench_safety.j2")
                ],
            }
        ],
    )
    scenario = scenario.with_annotations(
        {"behavior": behavior, "functional_category": functional_category}
    )
    return scenario.with_tags(tags) if tags else scenario


def _interact_of(scenario: AnyScenario) -> Interact[Any, Any, Any]:
    interact = scenario.steps[0].interacts[0]
    assert isinstance(interact, Interact)
    return interact


def _generator_of(scenario: AnyScenario) -> LLMGenerator[Any]:
    inputs = _interact_of(scenario).inputs
    assert isinstance(inputs, LLMGenerator)
    return inputs


def _load(
    monkeypatch: pytest.MonkeyPatch,
    generator: FramingScenarioGenerator,
    base: list[AnyScenario],
) -> list[AnyScenario]:
    """Run ``generator.load_scenarios`` over fixed base scenarios, no network.

    Patches the HuggingFace parent's load_scenarios to return the base
    scenarios so the framing transform is exercised in isolation.
    """
    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator,
        "load_scenarios",
        lambda self, description, languages: list(base),
    )
    return generator.load_scenarios("desc", ["en"])


@pytest.mark.parametrize("cls,prefix,template,max_steps", _VARIANTS)
def test_defaults_point_at_harmbench(cls, prefix, template, max_steps):
    gen = cls()
    assert gen.repo_id == "giskardai/harmbench-scenarios"
    assert gen.allow_commercial_use is True
    assert gen.name_prefix == prefix
    assert gen.prompt_path == f"giskard.scan::scenarios/{template}"
    assert gen.max_turns == max_steps


@pytest.mark.parametrize("cls,prefix,template,max_steps", _VARIANTS)
def test_dataset_prompt_replaced_by_framing_generator(
    monkeypatch, cls, prefix, template, max_steps
):
    scenarios = _load(monkeypatch, cls(), [_base_scenario("PAYLOAD_A")])

    assert len(scenarios) == 1
    interact = _interact_of(scenarios[0])
    assert isinstance(interact.inputs, LLMGenerator)
    assert interact.inputs.prompt_path == f"giskard.scan::scenarios/{template}"
    assert interact.inputs.max_steps == max_steps


@pytest.mark.parametrize("cls,prefix,template,max_steps", _VARIANTS)
def test_names_are_prefixed(monkeypatch, cls, prefix, template, max_steps):
    scenarios = _load(monkeypatch, cls(), [_base_scenario("P", name="HarmBench #0")])
    assert [s.name for s in scenarios] == [f"{prefix} - HarmBench #0"]


def test_only_standard_behaviors_are_framed(monkeypatch):
    base = [
        _base_scenario("STD", name="std", functional_category="standard"),
        _base_scenario("CP", name="cp", functional_category="copyright"),
        _base_scenario("CX", name="cx", functional_category="contextual"),
    ]
    scenarios = _load(monkeypatch, GrandmaFramingScenarioGenerator(), base)

    assert [s.annotations["behavior"] for s in scenarios] == ["STD"]


def test_behavior_without_category_is_kept(monkeypatch):
    base = _base_scenario("NOCAT", name="nocat")
    # drop the functional_category annotation
    base = base.with_annotations({"behavior": "NOCAT"})
    scenarios = _load(monkeypatch, GrandmaFramingScenarioGenerator(), [base])

    assert [s.annotations["behavior"] for s in scenarios] == ["NOCAT"]


def test_requested_languages_override_behavior_subset(monkeypatch):
    # The behavior is sourced from English, but the framing generator must
    # translate into the agent's languages, so the annotation the template reads
    # must carry the requested languages, not the English load subset.
    captured: dict[str, Any] = {}

    def fake_load(self, description, languages):
        captured["load_languages"] = languages
        return [_base_scenario("P")]

    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator, "load_scenarios", fake_load
    )
    scenarios = GrandmaFramingScenarioGenerator().load_scenarios("desc", ["fr", "es"])

    # behaviors are loaded from the English subset
    assert captured["load_languages"] == ["en"]
    # but the framed scenario advertises the agent's requested languages
    assert scenarios[0].annotations["languages"] == ["fr", "es"]


def test_behavior_annotation_is_preserved(monkeypatch):
    scenarios = _load(
        monkeypatch, MathFramingScenarioGenerator(), [_base_scenario("PAYLOAD_A")]
    )
    # the framing template reads trace.annotations.behavior, so it must survive
    assert scenarios[0].annotations["behavior"] == "PAYLOAD_A"


def test_safety_judge_is_preserved(monkeypatch):
    scenarios = _load(
        monkeypatch, CitationFramingScenarioGenerator(), [_base_scenario("P")]
    )
    # a successful framing attack is still scored by the HarmBench safety judge
    assert [type(c) for c in scenarios[0].steps[0].checks] == [LLMJudge]


def test_dataset_tags_are_preserved_and_framing_tags_added(monkeypatch):
    base = _base_scenario(
        "P",
        tags=["dataset:harmbench", "threat-type:harmful-content-generation"],
    )
    scenarios = _load(monkeypatch, GrandmaFramingScenarioGenerator(), [base])

    for s in scenarios:
        # dataset tags survive (with_tags() overwrites, so framing must append)
        assert "dataset:harmbench" in s.tags
        assert "threat-type:harmful-content-generation" in s.tags
        # framing tags are added
        assert "threat-type:prompt-injection" in s.tags
        assert "owasp:llm-top-10-2025:LLM01" in s.tags


def test_framed_interact_drives_the_framing_generator(monkeypatch):
    # The framing generator must replace the dataset prompt so the runtime
    # resolves the framing LLMGenerator (not the raw HarmBench prompt). The
    # private provider is rebuilt lazily from this field, so checking the field
    # is sufficient; runtime use of the live field is covered in giskard-checks.
    scenarios = _load(
        monkeypatch, GrandmaFramingScenarioGenerator(), [_base_scenario("RAW_PAYLOAD")]
    )
    interact = _interact_of(scenarios[0])
    assert isinstance(interact.inputs, LLMGenerator)
    assert not isinstance(interact.inputs, DatasetInputGenerator)


def test_empty_base_yields_nothing(monkeypatch):
    assert _load(monkeypatch, GrandmaFramingScenarioGenerator(), []) == []


def test_rejects_non_dataset_inputs(monkeypatch):
    bad = Scenario(
        name="bad",
        steps=[{"interacts": [Interact(inputs="hello")], "checks": []}],
    ).with_annotations({"behavior": "x", "functional_category": "standard"})
    with pytest.raises(TypeError, match="DatasetInputGenerator"):
        _load(monkeypatch, GrandmaFramingScenarioGenerator(), [bad])


async def test_likert_skipped_for_singleturn(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator,
        "load_scenarios",
        lambda self, description, languages: [_base_scenario("P")],
    )
    context = ScenarioContext(description="desc", languages=["en"])
    scenarios = await LikertFramingScenarioGenerator().generate_scenario(
        context, target_mode="singleturn"
    )
    assert scenarios == []


async def test_likert_runs_multiturn(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator,
        "load_scenarios",
        lambda self, description, languages: [_base_scenario("P")],
    )
    context = ScenarioContext(description="desc", languages=["en"])
    scenarios = await LikertFramingScenarioGenerator().generate_scenario(
        context, target_mode="multiturn"
    )
    assert len(scenarios) == 1
    assert _generator_of(scenarios[0]).max_steps == 2


async def test_singleturn_caps_generator_to_one_step(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceDatasetScenarioGenerator,
        "load_scenarios",
        lambda self, description, languages: [_base_scenario("P")],
    )
    context = ScenarioContext(description="desc", languages=["en"])
    scenarios = await GrandmaFramingScenarioGenerator().generate_scenario(
        context, target_mode="singleturn"
    )
    assert _generator_of(scenarios[0]).max_steps == 1
