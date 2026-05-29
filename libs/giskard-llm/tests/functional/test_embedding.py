"""Functional tests for embedding scenarios across providers."""

import os

import pytest
from giskard.llm import LLMClient
from giskard.llm.errors import UnsupportedOperationError

from .helpers import azure_foundry_v1_base_url

pytestmark = pytest.mark.functional

# -- Provider parametrization -------------------------------------------------

_EMBEDDING_MODELS = {
    "openai": os.getenv("TEST_OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    "google": os.getenv("TEST_GOOGLE_EMBEDDING_MODEL", "google/gemini-embedding-001"),
    "azure": os.getenv("TEST_AZURE_EMBEDDING_MODEL", "azure/text-embedding-3-small"),
    "azure_ai": os.getenv(
        "TEST_AZURE_AI_EMBEDDING_MODEL", "azure_ai/text-embedding-3-small"
    ),
    "azure_foundry_v1": os.getenv(
        "TEST_AZURE_FOUNDRY_V1_EMBEDDING_MODEL",
        "azure_foundry_v1/text-embedding-3-small",
    ),
}

_CONFIGURE_PARAMS = {  # pragma: allowlist secret
    "openai": {"provider": "openai", "api_key": "os.environ/OPENAI_API_KEY"},
    "google": {"provider": "google", "api_key": "os.environ/GOOGLE_API_KEY"},
    "azure": {
        "provider": "azure",
        "api_key": "os.environ/AZURE_API_KEY",
        "base_url": "os.environ/AZURE_API_BASE",
        "api_version": "os.environ/AZURE_API_VERSION",
    },
    "azure_ai": {
        "provider": "azure_ai",
        "api_key": "os.environ/AZURE_AI_API_KEY",
        "base_url": "os.environ/AZURE_AI_ENDPOINT",
    },
    "azure_foundry_v1": {
        "provider": "openai",
        "api_key": "os.environ/AZURE_AI_API_KEY",  # pragma: allowlist secret
        "base_url": azure_foundry_v1_base_url(),
    },
}

_PROVIDER_MARKS = {
    "openai": pytest.mark.openai,
    "google": pytest.mark.google,
    "azure": pytest.mark.azure,
    "azure_ai": pytest.mark.azure_ai,
    "azure_foundry_v1": pytest.mark.azure_foundry_v1,
}


def _make_client(provider: str) -> tuple[LLMClient, str]:
    client = LLMClient()
    alias = f"test-{provider}"
    client.configure(alias, **_CONFIGURE_PARAMS[provider])
    _, model_name = _EMBEDDING_MODELS[provider].split("/", 1)
    return client, f"{alias}/{model_name}"


_PROVIDER_PARAMS = [
    pytest.param(provider, marks=_PROVIDER_MARKS[provider], id=provider)
    for provider in _EMBEDDING_MODELS
]


# -- Embedding scenarios ------------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_basic_embedding(provider: str):
    """Single text input -> non-empty float list."""
    client, model = _make_client(provider)
    resp = await client.aembedding(model, ["Hello world"])
    assert len(resp.data) == 1
    assert len(resp.data[0].embedding) > 0
    assert all(isinstance(v, float) for v in resp.data[0].embedding)


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_embedding_dimensions(provider: str):
    """Explicit dimensions param -> embedding length matches."""
    client, model = _make_client(provider)
    resp = await client.aembedding(model, ["Hello world"], dimensions=64)
    assert len(resp.data) == 1
    assert len(resp.data[0].embedding) == 64


@pytest.mark.anthropic
async def test_embedding_unsupported_anthropic():
    """Anthropic -> appropriate error for embeddings."""
    client = LLMClient()
    client.configure(
        "test-anthropic",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    with pytest.raises(UnsupportedOperationError, match="does not support"):
        await client.aembedding("test-anthropic/claude-3", ["hello"])
