import pytest
from giskard.llm.errors import (
    AuthenticationError,
    LLMError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
)
from giskard.llm.providers.anthropic import AnthropicProvider
from giskard.llm.providers.azure_ai import AzureAIProvider
from giskard.llm.providers.google import GoogleProvider
from giskard.llm.providers.openai import OpenAIProvider


def test_llm_error_attributes():
    err = LLMError(429, "rate limited", "openai")
    assert err.status_code == 429
    assert err.message == "rate limited"
    assert err.provider == "openai"
    assert "[openai] 429" in str(err)


def test_subclass_hierarchy():
    err = RateLimitError(429, "slow down", "gemini")
    assert isinstance(err, LLMError)
    assert isinstance(err, RateLimitError)
    assert err.status_code == 429


@pytest.mark.no_providers
@pytest.mark.parametrize(
    "provider_cls, expected_hint",
    [
        (OpenAIProvider, "giskard-llm[openai]"),
        (AnthropicProvider, "giskard-llm[anthropic]"),
        (GoogleProvider, "giskard-llm[google]"),
        (AzureAIProvider, "giskard-llm[azure]"),
    ],
    ids=["openai", "anthropic", "google", "azure_ai"],
)
def test_provider_not_available_on_missing_sdk(provider_cls, expected_hint):
    """Instantiating a provider when its SDK is missing must raise
    ProviderNotAvailableError with the correct pip install hint."""
    with pytest.raises(ProviderNotAvailableError, match=expected_hint):
        provider_cls(api_key="dummy")


def test_error_chaining():
    original = ValueError("original error")
    try:
        raise ServerError(500, "internal", "openai") from original
    except LLMError as e:
        assert e.__cause__ is original
        assert e.status_code == 500


def test_all_error_types_are_llm_error():
    for cls in [AuthenticationError, RateLimitError, ServerError]:
        err = cls(400, "test", "test")
        assert isinstance(err, LLMError)
