from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.prompt_injection import (
    PromptInjectionScenarioGenerator,
)


async def test_prompt_injection_generator_loads_scenarios():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )
    assert len(scenarios) == 1  # LLM01 JSONL entry has 1 scenario


async def test_prompt_injection_generator_injects_annotations():
    gen = PromptInjectionScenarioGenerator()
    description = "A customer support chatbot"
    languages = ["en", "fr"]
    scenarios = await gen.generate_scenario(
        ScenarioContext(description=description, languages=languages)
    )
    for scenario in scenarios:
        assert scenario.annotations.get("description") == description
        assert set(scenario.annotations.get("languages", [])) == set(languages)


def _max_steps(scenario):
    return [
        interact.inputs.max_steps
        for step in scenario.steps
        for interact in step.interacts
        if hasattr(interact.inputs, "max_steps")
    ]


async def test_prompt_injection_multiturn_keeps_dataset_max_steps():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"]),
        target_mode="multiturn",
    )
    # The bundled LLM01 scenario encodes a multi-step interaction.
    assert any(steps > 1 for scenario in scenarios for steps in _max_steps(scenario))


async def test_prompt_injection_singleturn_caps_max_steps_to_1():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"]),
        target_mode="singleturn",
    )
    all_steps = [steps for scenario in scenarios for steps in _max_steps(scenario)]
    assert all_steps  # sanity: the dataset has interaction generators
    assert all(steps == 1 for steps in all_steps)
