from giskard.scan import (
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
    quality_scan,
    quality_suite_generator_registry,
    vulnerability_scan,
    vulnerability_suite_generator_registry,
)


def test_all_public_symbols_importable():
    assert callable(generate_suite)
    assert Document(content="doc").content == "doc"
    assert KnowledgeBase.from_texts(["doc"]).documents[0].content == "doc"
    assert callable(quality_scan)
    assert isinstance(quality_suite_generator_registry, SuiteGeneratorRegistry)
    assert callable(vulnerability_scan)
    assert isinstance(vulnerability_suite_generator_registry, SuiteGeneratorRegistry)


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
    assert sig.parameters["target_mode"].default == "multiturn"


def test_quality_scan_accepts_target_mode_param():
    """quality_scan must expose target_mode, mirroring vulnerability_scan."""
    import inspect

    from giskard.scan import quality_scan

    sig = inspect.signature(quality_scan)
    assert "target_mode" in sig.parameters
    assert sig.parameters["target_mode"].default == "multiturn"


def test_quality_suite_generator_registry_contains_builtin_generators():
    types = {type(g) for g in quality_suite_generator_registry.generators()}
    assert HallucinationScenarioGenerator in types
    assert KnowledgeBaseScenarioGenerator not in types
    assert MultiTopicScenarioGenerator in types
    assert OutOfScopeScenarioGenerator in types
    assert SplitQuestionsScenarioGenerator in types
    assert SycophancyScenarioGenerator in types
