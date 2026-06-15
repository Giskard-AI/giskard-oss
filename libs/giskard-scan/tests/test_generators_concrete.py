from giskard.scan.generators.prompt_injection import PromptInjectionScenarioGenerator


def test_prompt_injection_generator_is_importable():
    gen = PromptInjectionScenarioGenerator()
    assert gen.dataset_name == "prompt_injection"


async def test_prompt_injection_scenarios_have_tags():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario("Support agent", languages=["en"])
    assert len(scenarios) >= 1
    assert all(
        "threat-type:prompt-injection" in scenario.tags for scenario in scenarios
    )
