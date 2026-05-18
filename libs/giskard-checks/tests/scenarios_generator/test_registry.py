from typing import Any

import pytest
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios_generator.base import ScenarioGenerator
from giskard.checks.scenarios_generator.registry import SuiteGeneratorRegistry


class _GenA(ScenarioGenerator):
    async def generate_scenario(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        return []


class _GenB(ScenarioGenerator):
    value: int = 1

    async def generate_scenario(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        return []


# --- register ---


def test_register_bare_type_adds_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    assert len(registry.list()) == 1
    assert isinstance(registry.list()[0], _GenA)


def test_register_instance_adds_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA())
    assert len(registry.list()) == 1


def test_register_bare_type_equivalent_to_default_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    assert registry.list()[0] == _GenA()


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
    assert len(registry.list()) == 2


def test_register_bare_type_same_as_default_param_raises():
    # _GenB() == _GenB(value=1) since value defaults to 1
    registry = SuiteGeneratorRegistry()
    registry.register(_GenB)
    with pytest.raises(ValueError, match="_GenB"):
        registry.register(_GenB(value=1))


# --- unregister ---


def test_unregister_bare_type_removes_instance():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    registry.unregister(_GenA)
    assert registry.list() == []


def test_unregister_instance_removes_it():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA())
    registry.unregister(_GenA())
    assert registry.list() == []


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
    assert registry.list() == []


def test_clear_on_empty_registry_is_noop():
    registry = SuiteGeneratorRegistry()
    registry.clear()
    assert registry.list() == []


# --- list ---


def test_list_returns_copy():
    registry = SuiteGeneratorRegistry()
    registry.register(_GenA)
    snapshot = registry.list()
    snapshot.clear()
    assert len(registry.list()) == 1


# --- built-in generators ---


def test_suite_generator_registry_contains_builtin_generators():
    from giskard.checks.scenarios_generator.adversarial_generator import (
        AdversarialScenarioGenerator,
    )
    from giskard.checks.scenarios_generator.prompt_injection import (
        PromptInjectionScenarioGenerator,
    )
    from giskard.checks.scenarios_generator.registry import suite_generator_registry

    types = {type(g) for g in suite_generator_registry.list()}
    assert AdversarialScenarioGenerator in types
    assert PromptInjectionScenarioGenerator in types
