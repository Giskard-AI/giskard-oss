from giskard.checks import Trace
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


async def test_prompt_injection_category_is_reachable_at_runtime():
    """The dataset's ``category`` must survive parsing and reach the trace.

    ``category`` lives inside the scenario's ``annotations`` object in the
    JSONL. A top-level ``category`` key would be silently dropped by pydantic,
    making it unreachable at runtime.
    """
    gen = PromptInjectionScenarioGenerator()
    description = "A documentation chatbot"
    languages = ["en"]
    scenarios = await gen.generate_scenario(
        ScenarioContext(description=description, languages=languages)
    )
    assert scenarios
    for scenario in scenarios:
        # Reachable on the parsed model...
        assert scenario.annotations.get("category") == "llm01_indirect_injection"

        # ...and at runtime: the runner seeds the trace from the scenario's
        # annotations. Drop the LLM-backed steps so the scenario can run
        # offline while keeping the annotations under test.
        scenario.steps = []
        result = await scenario.run(multiple_runs=1)
        trace: Trace[object, object] = result.final_trace
        assert trace.annotations["category"] == "llm01_indirect_injection"
        # The loader-injected annotations are not clobbered.
        assert trace.annotations["description"] == description
        assert trace.annotations["languages"] == languages


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
