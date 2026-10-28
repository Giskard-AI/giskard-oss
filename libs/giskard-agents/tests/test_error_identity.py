"""Regression: agents.Error must be the same type as core.Error."""

from giskard.agents import Error as AgentsError
from giskard.agents.errors.serializable import Error as SerializableError
from giskard.core import Error as CoreError


def test_agents_error_is_core_error():
    """Public import paths must resolve to one canonical Error type."""
    assert AgentsError is CoreError
    assert SerializableError is CoreError


def test_isinstance_and_model_validate_agree_across_imports():
    """isinstance and model_validate must agree regardless of import path."""
    agents_error = AgentsError(message="Something went wrong")
    core_error = CoreError.model_validate(agents_error.model_dump())

    assert isinstance(agents_error, CoreError)
    assert isinstance(core_error, AgentsError)
    assert AgentsError.model_validate(core_error.model_dump()) == agents_error
