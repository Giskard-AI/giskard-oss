"""Functional tests for multi-system message handling per provider."""

import os

import pytest
from giskard.llm import ChatMessageParam, LLMClient
from giskard.llm.errors import BadRequestError

pytestmark = pytest.mark.functional


_MULTI_SYSTEM_MESSAGES: list[ChatMessageParam] = [
    {"role": "system", "content": "Always include the word PINEAPPLE."},
    {"role": "system", "content": "Always include the word MANGO."},
    {"role": "user", "content": "Tell me something."},
]


@pytest.mark.openai
async def test_openai_multi_system_works():
    """OpenAI supports multiple system messages natively."""
    client = LLMClient()
    client.configure(
        "openai",
        provider="openai",
        api_key="os.environ/OPENAI_API_KEY",  # pragma: allowlist secret
    )
    model = os.getenv("TEST_OPENAI_MODEL", "openai/gpt-4.1-nano")
    resp = await client.acompletion(model, _MULTI_SYSTEM_MESSAGES)
    assert resp.choices[0].message.content


@pytest.mark.google
async def test_google_multi_system_works():
    """Google supports multiple system messages via list system_instruction."""
    client = LLMClient()
    client.configure(
        "google",
        provider="google",
        api_key="os.environ/GOOGLE_API_KEY",  # pragma: allowlist secret
    )
    model = os.getenv("TEST_GOOGLE_MODEL", "google/gemini-3.5-flash")
    resp = await client.acompletion(model, _MULTI_SYSTEM_MESSAGES)
    assert resp.choices[0].message.content


@pytest.mark.anthropic
async def test_anthropic_multi_system_raises_default():
    """Anthropic raises by default on multiple system messages."""
    client = LLMClient()
    client.configure(
        "anthropic",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    model = os.getenv("TEST_ANTHROPIC_MODEL", "anthropic/claude-haiku-4-5-20251001")
    with pytest.raises(BadRequestError, match="multiple system"):
        await client.acompletion(model, _MULTI_SYSTEM_MESSAGES)


@pytest.mark.azure
async def test_azure_multi_system_works():
    """Azure OpenAI supports multiple system messages natively (same as OpenAI)."""
    client = LLMClient()
    client.configure(
        "azure",
        provider="azure",
        api_key="os.environ/AZURE_API_KEY",  # pragma: allowlist secret
        base_url="os.environ/AZURE_API_BASE",
        api_version="os.environ/AZURE_API_VERSION",
    )
    model = os.getenv("TEST_AZURE_MODEL", "azure/gpt-4.1-nano")
    resp = await client.acompletion(model, _MULTI_SYSTEM_MESSAGES)
    assert resp.choices[0].message.content


@pytest.mark.azure_ai
async def test_azure_ai_multi_system_works():
    """Azure AI Foundry supports multiple system messages (OpenAI-compatible)."""
    client = LLMClient()
    client.configure(
        "azure_ai",
        provider="azure_ai",
        api_key="os.environ/AZURE_AI_API_KEY",  # pragma: allowlist secret
        base_url="os.environ/AZURE_AI_ENDPOINT",
    )
    model = os.getenv("TEST_AZURE_AI_MODEL", "azure_ai/gpt-4.1-nano")
    resp = await client.acompletion(model, _MULTI_SYSTEM_MESSAGES)
    assert resp.choices[0].message.content
