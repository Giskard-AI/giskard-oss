"""Smoke tests for import safety without optional SDK dependencies.

- The first two tests run on every install: they verify pure-Python code with
  no optional dependency.
- The third test uses pytest.skipif to guard against litellm being present;
  it verifies that importing the LiteLLM generator module does not raise even
  when litellm is absent (import is guarded inside the module).
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
    # The LiteLLMGenerator module guards its litellm import inside the class body,
    # so importing the module itself must not raise even without litellm installed.
    from giskard.agents.generators import litellm_generator  # noqa: F401
