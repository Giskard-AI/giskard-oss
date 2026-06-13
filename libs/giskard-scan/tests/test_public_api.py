from giskard.scan import (
    AdversarialScenarioGenerator,
    CrescendoAttackScenarioGenerator,
    GOATAttackScenarioGenerator,
    PromptInjectionScenarioGenerator,
    SuiteGeneratorRegistry,
    generate_suite,
    vulnerability_scan,
    vulnerability_suite_generator_registry,
)


def test_all_public_symbols_importable():
    assert callable(generate_suite)
    assert callable(vulnerability_scan)
    assert isinstance(vulnerability_suite_generator_registry, SuiteGeneratorRegistry)


def test_vulnerability_suite_generator_registry_contains_builtin_generators():
    types = {type(g) for g in vulnerability_suite_generator_registry.generators()}
    assert AdversarialScenarioGenerator in types
    assert CrescendoAttackScenarioGenerator in types
    assert GOATAttackScenarioGenerator in types
    assert PromptInjectionScenarioGenerator in types
