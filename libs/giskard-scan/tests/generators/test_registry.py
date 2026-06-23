from typing import Any

import numpy as np
import pytest
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.scan.generators.base import ScenarioContext, ScenarioGenerator
from giskard.scan.registry import SuiteGeneratorRegistry


class _GenA(ScenarioGenerator):
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        return []


class _GenB(ScenarioGenerator):
    value: int = 1

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        return []


# --- register ---


def test_register_bare_type_adds_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    assert len(registry.generators()) == 1
    assert isinstance(registry.generators()[0], _GenA)


def test_register_instance_adds_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA())
    assert len(registry.generators()) == 1


def test_register_bare_type_equivalent_to_default_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    assert registry.generators()[0] == _GenA()


def test_register_duplicate_bare_type_raises():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    with pytest.raises(ValueError, match="_GenA"):
        registry.register(_GenA)


def test_register_duplicate_instance_raises():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA())
    with pytest.raises(ValueError, match="_GenA"):
        registry.register(_GenA())


def test_register_different_parameterized_instances_both_succeed():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenB(value=1))
    registry.register(_GenB(value=2))
    assert len(registry.generators()) == 2


def test_register_bare_type_same_as_default_param_raises():
    # _GenB() == _GenB(value=1) since value defaults to 1
    registry = SuiteGeneratorRegistry()
    registry.register(_GenB)
    with pytest.raises(ValueError, match="_GenB"):
        registry.register(_GenB(value=1))


def test_register_non_generator_type_raises():
    registry = SuiteGeneratorRegistry()
    with pytest.raises(TypeError, match="ScenarioGenerator"):
        registry.register(int)  # pyright: ignore[reportArgumentType]


# --- unregister ---


def test_unregister_bare_type_removes_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    registry.unregister(_GenA)
    assert registry.generators() == []


def test_unregister_instance_removes_it():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA())
    registry.unregister(_GenA())
    assert registry.generators() == []


def test_unregister_not_registered_raises():
    registry = SuiteGeneratorRegistry()
    with pytest.raises(ValueError, match="_GenA"):
        registry.unregister(_GenA)


# --- clear ---


def test_clear_empties_registry():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    registry.register(_GenB)
    registry.clear()
    assert registry.generators() == []


def test_clear_on_empty_registry_is_noop():
    registry = SuiteGeneratorRegistry()
    registry.clear()
    assert registry.generators() == []


# --- generators ---


def test_generators_returns_copy():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    snapshot = registry.generators()
    snapshot.clear()
    assert len(registry.generators()) == 1
