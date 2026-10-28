import giskard.scan as scan
import giskard.scan.generators as generators
from giskard.scan import (
    DEFAULT_TARGET_MODE,
    AdversarialScenarioGenerator,
    CrescendoAttackScenarioGenerator,
    Document,
    GOATAttackScenarioGenerator,
    HallucinationScenarioGenerator,
    KnowledgeBase,
    KnowledgeBaseScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
    PromptInjectionScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    SuiteGeneratorRegistry,
    SycophancyScenarioGenerator,
    generate_suite,
    list_scan_items,
    quality_scan,
    quality_suite_generator_registry,
    third_party_scan,
    vulnerability_scan,
    vulnerability_suite_generator_registry,
)


def _registry_generator_types(registry: SuiteGeneratorRegistry) -> set[type]:
    return {type(generator) for generator in registry.generators()}


def test_default_registry_generators_are_public():
    generator_types = _registry_generator_types(
        vulnerability_suite_generator_registry
    ) | _registry_generator_types(quality_suite_generator_registry)
    for module in (scan, generators):
        public_names = set(module.__all__)
        for generator_type in generator_types:
            assert generator_type.__name__ in public_names
            assert getattr(module, generator_type.__name__) is generator_type


def test_all_public_symbols_importable():
    assert callable(generate_suite)
    assert Document(content="doc").content == "doc"
    assert KnowledgeBase.from_texts(["doc"]).documents[0].content == "doc"
    assert callable(quality_scan)
    assert isinstance(quality_suite_generator_registry, SuiteGeneratorRegistry)
    assert callable(vulnerability_scan)
    assert isinstance(vulnerability_suite_generator_registry, SuiteGeneratorRegistry)
    assert callable(third_party_scan)
    assert callable(list_scan_items)
    assert DEFAULT_TARGET_MODE == "multiturn"


def test_vulnerability_suite_generator_registry_contains_builtin_generators():
    types = {type(g) for g in vulnerability_suite_generator_registry.generators()}
    assert AdversarialScenarioGenerator in types
    assert CrescendoAttackScenarioGenerator in types
    assert GOATAttackScenarioGenerator in types
    assert PromptInjectionScenarioGenerator in types


def test_vulnerability_scan_accepts_target_mode_param():
    """vulnerability_scan signature must include target_mode parameter."""
    import inspect

    from giskard.scan import vulnerability_scan

    sig = inspect.signature(vulnerability_scan)
    assert "target_mode" in sig.parameters
    assert sig.parameters["target_mode"].default == DEFAULT_TARGET_MODE


def test_quality_scan_accepts_target_mode_param():
    """quality_scan must expose target_mode, mirroring vulnerability_scan."""
    import inspect

    from giskard.scan import quality_scan

    sig = inspect.signature(quality_scan)
    assert "target_mode" in sig.parameters
    assert sig.parameters["target_mode"].default == DEFAULT_TARGET_MODE


def test_quality_suite_generator_registry_contains_builtin_generators():
    types = {type(g) for g in quality_suite_generator_registry.generators()}
    assert HallucinationScenarioGenerator in types
    assert KnowledgeBaseScenarioGenerator not in types
    assert MultiTopicScenarioGenerator in types
    assert OutOfScopeScenarioGenerator in types
    assert SplitQuestionsScenarioGenerator in types
    assert SycophancyScenarioGenerator in types


def test_list_scan_items_giskard():
    names = list_scan_items("giskard")
    assert "AdversarialScenarioGenerator" in names
    assert "HallucinationScenarioGenerator" in names
