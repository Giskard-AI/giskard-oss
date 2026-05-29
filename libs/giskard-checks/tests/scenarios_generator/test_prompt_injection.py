from giskard.checks.scenarios_generator.prompt_injection import (
    PromptInjectionScenarioGenerator,
)


async def test_prompt_injection_generator_loads_scenarios():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        description="A documentation chatbot", languages=["en"]
    )
    assert len(scenarios) == 1  # LLM01 JSONL entry has 1 scenario


async def test_prompt_injection_generator_injects_annotations():
    gen = PromptInjectionScenarioGenerator()
    description = "A customer support chatbot"
    languages = ["en", "fr"]
    scenarios = await gen.generate_scenario(
        description=description, languages=languages
    )
    for scenario in scenarios:
        assert scenario.annotations.get("description") == description
        assert set(scenario.annotations.get("languages", [])) == set(languages)
