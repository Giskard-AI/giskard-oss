import numpy as np
from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import LLMJudge
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.crescendo import (
    DEFAULT_CRESCENDO_MAX_TURNS,
    DEFAULT_CRESCENDO_OBJECTIVES,
    CrescendoAttackScenarioGenerator,
)


async def test_crescendo_generator_returns_default_objectives():
    gen = CrescendoAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"])
    )

    assert len(scenarios) == len(DEFAULT_CRESCENDO_OBJECTIVES)
    assert scenarios[0].annotations["objective_name"] == "botulinum toxin"
    assert (
        scenarios[0].annotations["goal"]
        == DEFAULT_CRESCENDO_OBJECTIVES["botulinum toxin"]
    )


async def test_crescendo_generator_budget_subsamples_reproducibly():
    gen = CrescendoAttackScenarioGenerator()

    first = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en", "fr"]),
        max_scenarios=3,
        rng=np.random.default_rng(42),
    )
    second = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en", "fr"]),
        max_scenarios=3,
        rng=np.random.default_rng(42),
    )

    assert len(first) == 3
    assert [scenario.name for scenario in first] == [
        scenario.name for scenario in second
    ]
    assert [scenario.annotations["language"] for scenario in first] == [
        scenario.annotations["language"] for scenario in second
    ]


async def test_crescendo_generator_builds_multiturn_scenario_with_annotations():
    gen = CrescendoAttackScenarioGenerator(max_turns=4)
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en", "fr"])
    )
    scenario = scenarios[0]

    assert scenario.annotations["description"] == "A safety chatbot"
    assert scenario.annotations["language"] in {"en", "fr"}
    assert scenario.annotations["objective_name"] == "botulinum toxin"
    assert scenario.annotations["max_turns"] == 4

    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert interaction.inputs.max_steps == 4
    assert (
        interaction.inputs.prompt_path == "giskard.scan::scenarios/crescendo_attack.j2"
    )

    check = scenario.steps[0].checks[0]
    assert isinstance(check, LLMJudge)
    assert check.prompt_path == "giskard.scan::judges/crescendo_eval.j2"
    assert scenario.name == "Crescendo Attack - botulinum toxin"


def test_crescendo_generator_uses_default_max_turns():
    assert CrescendoAttackScenarioGenerator().max_turns == DEFAULT_CRESCENDO_MAX_TURNS


async def test_crescendo_singleturn_returns_empty():
    gen = CrescendoAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"]),
        target_mode="singleturn",
    )
    assert scenarios == []


async def test_crescendo_multiturn_still_returns_scenarios():
    gen = CrescendoAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"]),
        target_mode="multiturn",
    )
    assert len(scenarios) == len(DEFAULT_CRESCENDO_OBJECTIVES)


async def test_crescendo_generator_falls_back_to_english_with_empty_languages():
    """An empty `languages` list should fall back to English, mirroring
    KnowledgeBaseScenarioGenerator._sample_languages, instead of crashing in
    `rng.integers(0)`."""
    gen = CrescendoAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=[])
    )

    assert len(scenarios) == len(DEFAULT_CRESCENDO_OBJECTIVES)
    assert all(scenario.annotations["language"] == "en" for scenario in scenarios)
