"""Helpers shared by functional provider tests."""

import os

import pytest

ANTHROPIC_PROVIDER_PARAMS = [
    pytest.param("anthropic", marks=pytest.mark.anthropic, id="anthropic"),
    pytest.param(
        "azure_anthropic", marks=pytest.mark.azure_anthropic, id="azure_anthropic"
    ),
]


def anthropic_configure_params(provider: str) -> dict[str, str | None]:
    """Return ``LLMClient.configure`` kwargs for Anthropic.com or Foundry Anthropic."""
    if provider == "azure_anthropic":
        return {
            "provider": "anthropic",
            "api_key": "os.environ/AZURE_AI_API_KEY",  # pragma: allowlist secret
            "base_url": azure_foundry_anthropic_base_url(),
        }
    return {
        "provider": "anthropic",
        "api_key": "os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    }


def azure_foundry_anthropic_base_url() -> str | None:
    """Return the Anthropic Messages base URL on Azure AI Foundry.

    Foundry serves Claude at ``{AZURE_AI_ENDPOINT}/anthropic``. The Anthropic
    SDK appends ``/v1/messages``.
    """
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    if not endpoint:
        return None
    return f"{endpoint.rstrip('/')}/anthropic"


def azure_foundry_v1_base_url() -> str | None:
    """Return the OpenAI-compatible Azure Foundry v1 base URL.

    The CI environment already exposes ``AZURE_AI_ENDPOINT`` for Azure AI
    Foundry tests. The OpenAI SDK expects the v1 base URL, so append
    ``/openai/v1/`` unless the caller supplied an explicit v1 endpoint.
    """
    explicit = os.getenv("AZURE_AI_OPENAI_V1_ENDPOINT")
    if explicit:
        return explicit

    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    if not endpoint:
        return None

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/openai/v1"):
        return f"{normalized}/"
    return f"{normalized}/openai/v1/"
