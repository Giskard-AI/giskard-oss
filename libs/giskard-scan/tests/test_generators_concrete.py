from giskard.scan.generators.prompt_injection import PromptInjectionScenarioGenerator


def test_prompt_injection_generator_is_importable():
    gen = PromptInjectionScenarioGenerator()
    assert gen.dataset_name == "prompt_injection"


def test_prompt_injection_generator_has_tags():
    gen = PromptInjectionScenarioGenerator()
    assert any("prompt-injection" in t for t in gen.tags)
