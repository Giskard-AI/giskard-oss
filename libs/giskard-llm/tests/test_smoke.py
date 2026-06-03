"""Smoke tests for import safety and lazy SDK loading.

- The first three tests run on every install (no skip condition): they only
  exercise pure-Python code with no optional dependency.
- The last three use pytest.skipif to guard against the specific SDK being
  present; they verify ProviderNotAvailableError is raised when that SDK is absent.
"""

import importlib.util

import pytest


def test_package_import_does_not_raise():
    import giskard.llm  # noqa: F401


def test_public_api_is_accessible():
    import giskard.llm as m

    for name in [
        "LLMClient",
        "acompletion",
        "aembedding",
        "aresponse",
        "configure",
        "reset",
        "LLMError",
        "ProviderNotAvailableError",
    ]:
        assert hasattr(m, name), f"giskard.llm missing attribute: {name}"


def test_provider_modules_import_without_raising():
    # Module-level import must not raise; SDK import is deferred to instantiation
    from giskard.llm.providers import anthropic, google, openai  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None,
    reason="openai is installed; ProviderNotAvailableError would not be raised",
)
def test_openai_provider_raises_not_available_on_instantiation():
    from giskard.llm.errors import ProviderNotAvailableError
    from giskard.llm.providers.openai import OpenAIProvider

    with pytest.raises(ProviderNotAvailableError, match="giskard-llm\\[openai\\]"):
        OpenAIProvider(api_key="dummy")  # pragma: allowlist secret


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None,
    reason="anthropic is installed; ProviderNotAvailableError would not be raised",
)
def test_anthropic_provider_raises_not_available_on_instantiation():
    from giskard.llm.errors import ProviderNotAvailableError
    from giskard.llm.providers.anthropic import AnthropicProvider

    with pytest.raises(ProviderNotAvailableError, match="giskard-llm\\[anthropic\\]"):
        AnthropicProvider(api_key="dummy")  # pragma: allowlist secret


@pytest.mark.skipif(
    importlib.util.find_spec("google.genai") is not None,
    reason="google-genai is installed; ProviderNotAvailableError would not be raised",
)
def test_google_provider_raises_not_available_on_instantiation():
    from giskard.llm.errors import ProviderNotAvailableError
    from giskard.llm.providers.google import GoogleProvider

    with pytest.raises(ProviderNotAvailableError, match="giskard-llm\\[google\\]"):
        GoogleProvider(api_key="dummy")  # pragma: allowlist secret
