from typing import Any

import numpy as np
import pytest
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios_generator.base import ScenarioGenerator
from giskard.checks.scenarios_generator.catalog import generate_suite
from giskard.checks.scenarios_generator.registry import suite_generator_registry


class _StubGenerator(ScenarioGenerator):
    """Returns a fixed single scenario for testing."""

    name: str = "stub"

    async def generate_scenario(
        self,
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        return [Scenario(name=f"stub-{self.name}")]


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot and restore registry around each test."""
    original = suite_generator_registry.generators()
    suite_generator_registry.clear()
    yield
    suite_generator_registry.clear()
    for g in original:
        suite_generator_registry.register(g)


async def test_generate_suite_uses_registry_by_default():
    suite_generator_registry.register(_StubGenerator(name="a"))
    suite = await generate_suite("My chatbot", languages=["en"])
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "stub-a"


async def test_generate_suite_generators_override_bypasses_registry():
    suite_generator_registry.register(_StubGenerator(name="registry"))
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator(name="override")],
    )
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "stub-override"


async def test_generate_suite_generators_bare_type_is_normalized():
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator],
    )
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "stub-stub"


async def test_generate_suite_empty_registry_returns_empty_suite():
    suite = await generate_suite("My chatbot", languages=["en"])
    assert suite.scenarios == []


async def test_generate_suite_empty_generators_override_returns_empty_suite():
    suite_generator_registry.register(_StubGenerator(name="a"))
    suite = await generate_suite("My chatbot", languages=["en"], generators=[])
    assert suite.scenarios == []


async def test_generate_suite_max_scenarios_limits_output():
    suite_generator_registry.register(_StubGenerator(name="a"))
    suite_generator_registry.register(_StubGenerator(name="b"))
    suite = await generate_suite("My chatbot", languages=["en"], max_scenarios=1)
    assert len(suite.scenarios) == 1
