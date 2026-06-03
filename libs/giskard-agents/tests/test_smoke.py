"""Smoke tests for import safety without optional SDK dependencies.

- The first two tests run on every install: they verify pure-Python code with
  no optional dependency.
- The last two use pytest.skipif to guard against litellm being present;
  they verify that the module imports safely and that instantiation raises
  ImportError with a helpful message when litellm is absent.
"""

import importlib.util

import pytest


def test_package_import_does_not_raise():
    import giskard.agents  # noqa: F401


def test_core_public_api_is_accessible():
    import giskard.agents as m

    for name in [
        "Generator",
        "ChatWorkflow",
    ]:
        assert hasattr(m, name), f"giskard.agents missing attribute: {name}"


@pytest.mark.skipif(
    importlib.util.find_spec("litellm") is not None,
    reason="litellm is installed; this test verifies behavior when it is absent",
)
def test_litellm_generator_module_import_does_not_raise():
    # After the lazy-import refactor, importing the module is safe even without litellm.
    from giskard.agents.generators import litellm_generator  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("litellm") is not None,
    reason="litellm is installed; InstantiationError would not be raised",
)
def test_litellm_generator_raises_import_error_on_instantiation():
    from giskard.agents.generators.litellm_generator import LiteLLMGenerator

    with pytest.raises(ImportError, match="giskard-agents\\[litellm\\]"):
        LiteLLMGenerator(model="gemini/gemini-3.5-flash")
