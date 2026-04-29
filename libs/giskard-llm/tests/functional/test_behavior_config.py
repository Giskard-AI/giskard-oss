"""Functional tests for behavior configuration via LLMClient.configure()."""

import os

import pytest
from giskard.llm import ChatMessageParam, LLMClient
from giskard.llm.errors import BadRequestError

pytestmark = [pytest.mark.functional, pytest.mark.anthropic]

_MODEL = os.getenv("TEST_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


async def test_anthropic_strict_multi_system_raises():
    """Default Anthropic config: multiple system messages -> BadRequestError."""
    client = LLMClient()
    client.configure(
        "anthropic-strict",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    with pytest.raises(BadRequestError, match="multiple system"):
        await client.acompletion(
            f"anthropic-strict/{_MODEL}",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hi"},
            ],
        )


async def test_anthropic_relaxed_multi_system_merges():
    """Anthropic with merge_system=True: multiple system messages -> valid response."""
    client = LLMClient()
    client.configure(
        "anthropic-relaxed",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
        merge_system=True,
    )
    resp = await client.acompletion(
        f"anthropic-relaxed/{_MODEL}",
        [
            {"role": "system", "content": "Always include the word PINEAPPLE."},
            {"role": "system", "content": "Always include the word MANGO."},
            {"role": "user", "content": "Tell me something."},
        ],
    )
    content = resp.choices[0].message.text
    assert isinstance(content, str)
    assert "pineapple" in content.lower() and "mango" in content.lower()


async def test_both_aliases_coexist():
    """Strict and relaxed aliases on the same client work independently."""
    client = LLMClient()
    client.configure(
        "strict",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    client.configure(
        "relaxed",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
        merge_system=True,
    )

    messages: list[ChatMessageParam] = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "Hi"},
    ]

    with pytest.raises(BadRequestError):
        await client.acompletion(f"strict/{_MODEL}", messages)

    resp = await client.acompletion(f"relaxed/{_MODEL}", messages)
    assert resp.choices[0].message.content
