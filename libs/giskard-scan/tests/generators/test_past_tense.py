import numpy as np
from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import LLMJudge
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.past_tense import (
    DEFAULT_PAST_TENSE_OBJECTIVES,
    PastTenseAttackScenarioGenerator,
)


async def test_past_tense_generator_returns_default_objectives():
    gen = PastTenseAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"])
    )

    assert len(scenarios) == len(DEFAULT_PAST_TENSE_OBJECTIVES)
    first_name = next(iter(DEFAULT_PAST_TENSE_OBJECTIVES))
    assert scenarios[0].annotations["objective_name"] == first_name
    assert scenarios[0].annotations["goal"] == DEFAULT_PAST_TENSE_OBJECTIVES[first_name]


async def test_past_tense_generator_budget_subsamples_reproducibly():
    gen = PastTenseAttackScenarioGenerator()

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
    assert [s.name for s in first] == [s.name for s in second]
    assert [s.annotations["language"] for s in first] == [
        s.annotations["language"] for s in second
    ]


async def test_past_tense_generator_builds_singleturn_scenario_with_annotations():
    gen = PastTenseAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en", "fr"])
    )
    scenario = scenarios[0]

    assert scenario.annotations["description"] == "A safety chatbot"
    assert scenario.annotations["language"] in {"en", "fr"}
    # the harmbench_safety judge grades against `behavior`
    assert scenario.annotations["behavior"] == scenario.annotations["goal"]

    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert interaction.inputs.max_steps == 1
    assert (
        interaction.inputs.prompt_path == "giskard.scan::scenarios/past_tense_attack.j2"
    )

    check = scenario.steps[0].checks[0]
    assert isinstance(check, LLMJudge)
    assert check.prompt_path == "giskard.scan::judges/harmbench_safety.j2"
    assert scenario.name.startswith("Past Tense Attack - ")


async def test_past_tense_singleturn_returns_scenarios():
    # Unlike multi-turn attacks, past-tense reformulation is single-turn.
    gen = PastTenseAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"]),
        target_mode="singleturn",
    )
    assert len(scenarios) == len(DEFAULT_PAST_TENSE_OBJECTIVES)


async def test_past_tense_multiturn_returns_scenarios():
    gen = PastTenseAttackScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A safety chatbot", languages=["en"]),
        target_mode="multiturn",
    )
    assert len(scenarios) == len(DEFAULT_PAST_TENSE_OBJECTIVES)
