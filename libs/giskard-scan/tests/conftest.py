import pytest
from giskard.scan.quality import quality_suite_generator_registry
from giskard.scan.vulnerability import vulnerability_suite_generator_registry


@pytest.fixture
def isolated_vulnerability_registry():
    """Snapshot and restore the vulnerability generator registry."""
    original = vulnerability_suite_generator_registry.generators()
    vulnerability_suite_generator_registry.clear()
    yield
    vulnerability_suite_generator_registry.clear()
    for generator in original:
        vulnerability_suite_generator_registry.register(generator)


@pytest.fixture
def isolated_quality_registry():
    """Snapshot and restore the quality generator registry."""
    original = quality_suite_generator_registry.generators()
    quality_suite_generator_registry.clear()
    yield
    quality_suite_generator_registry.clear()
    for generator in original:
        quality_suite_generator_registry.register(generator)
