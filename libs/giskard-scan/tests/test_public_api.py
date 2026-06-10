from giskard.scan import (
    SuiteGeneratorRegistry,
    generate_suite,
    suite_generator_registry,
)


def test_all_public_symbols_importable():
    assert callable(generate_suite)
    assert isinstance(suite_generator_registry, SuiteGeneratorRegistry)
