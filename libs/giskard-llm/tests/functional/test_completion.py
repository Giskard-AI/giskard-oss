"""Functional tests for completion scenarios across all providers.

Design principle: every test must work with the weakest model available.
Assert on structure (non-empty, correct role, correct type), not content.
"""

import json
import os

import pytest
from giskard.llm import LLMClient
from giskard.llm.errors import BadRequestError
from giskard.llm.types import ToolDefParam
from pydantic import BaseModel

pytestmark = pytest.mark.functional

# -- Provider parametrization -------------------------------------------------

_MODELS = {
    "openai": os.getenv("TEST_OPENAI_MODEL", "openai/gpt-4.1-nano"),
    "bare": os.getenv("TEST_BARE_MODEL", "gpt-4.1-nano"),
    "google": os.getenv("TEST_GOOGLE_MODEL", "google/gemini-2.0-flash"),
    "gemini": os.getenv("TEST_GEMINI_MODEL", "gemini/gemini-2.0-flash"),
    "anthropic": os.getenv(
        "TEST_ANTHROPIC_MODEL", "anthropic/claude-haiku-4-5-20251001"
    ),
    "azure": os.getenv("TEST_AZURE_MODEL", "azure/gpt-4.1-nano"),
    "azure_ai": os.getenv("TEST_AZURE_AI_MODEL", "azure_ai/gpt-4.1-nano"),
}

# `bare` intentionally has no entry: it exercises the no-prefix path where
# the default openai registry entry is used without any client.configure() call.
_CONFIGURE_PARAMS = {  # pragma: allowlist secret
    "openai": {"provider": "openai", "api_key": "os.environ/OPENAI_API_KEY"},
    "google": {"provider": "google", "api_key": "os.environ/GOOGLE_API_KEY"},
    "gemini": {"provider": "gemini", "api_key": "os.environ/GOOGLE_API_KEY"},
    "anthropic": {"provider": "anthropic", "api_key": "os.environ/ANTHROPIC_API_KEY"},
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
}

_PROVIDER_MARKS = {
    "openai": pytest.mark.openai,
    "bare": pytest.mark.bare,
    "google": pytest.mark.google,
    "gemini": pytest.mark.gemini,
    "anthropic": pytest.mark.anthropic,
    "azure": pytest.mark.azure,
    "azure_ai": pytest.mark.azure_ai,
}

_PROVIDER_PARAMS = [
    pytest.param(provider, marks=_PROVIDER_MARKS[provider], id=provider)
    for provider in _MODELS
]


def _make_client(provider: str) -> tuple[LLMClient, str]:
    """Create a configured LLMClient and return (client, model_string)."""
    client = LLMClient()
    if provider == "bare":
        # No configure(); LLMClient falls back to the default openai registry entry
        # and the SDK reads OPENAI_API_KEY from the environment.
        return client, _MODELS["bare"]
    alias = f"test-{provider}"
    client.configure(alias, **_CONFIGURE_PARAMS[provider])
    model = _MODELS[provider]
    _, model_name = model.split("/", 1)
    return client, f"{alias}/{model_name}"


# -- Message composition scenarios --------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_user_only(provider: str):
    """Single user message -> non-empty assistant response."""
    client, model = _make_client(provider)
    resp = await client.acompletion(model, [{"role": "user", "content": "Say hello"}])
    assert len(resp.choices) > 0
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.text
    assert isinstance(resp.choices[0].message.text, str)
    assert len(resp.choices[0].message.text.strip()) > 0


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_system_only_raises(provider: str):
    """System-only message -> BadRequestError before API call."""
    client, model = _make_client(provider)
    with pytest.raises(BadRequestError, match="non-system"):
        await client.acompletion(model, [{"role": "system", "content": "Be helpful"}])


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_system_user_keyword_injection(provider: str):
    """System message with keyword injection -> response contains keyword."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model,
        [
            {
                "role": "system",
                "content": "Always include the word PINEAPPLE in your response.",
            },
            {"role": "user", "content": "Tell me something."},
        ],
    )
    assert resp.choices[0].message.text
    assert isinstance(resp.choices[0].message.text, str)
    assert "pineapple" in resp.choices[0].message.text.lower()


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_multi_turn(provider: str):
    """System + user + assistant + user -> non-empty follow-up response."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And what is 3+3?"},
        ],
    )
    assert resp.choices[0].message.text
    assert isinstance(resp.choices[0].message.text, str)
    assert len(resp.choices[0].message.text.strip()) > 0


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_empty_messages_raises(provider: str):
    """Empty messages list -> BadRequestError."""
    client, model = _make_client(provider)
    with pytest.raises(BadRequestError, match="must not be empty"):
        await client.acompletion(model, [])


# -- Tool call scenarios ------------------------------------------------------


ADD_TOOL: ToolDefParam = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_tool_call(provider: str):
    """User asks a question with a tool -> tool_calls finish reason, parseable args."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model,
        [{"role": "user", "content": "What is 2+2? Use the add tool."}],
        tools=[ADD_TOOL],
    )
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls
    tc = choice.message.tool_calls[0]
    assert tc.function.name == "add"
    args = tc.function.arguments
    assert "a" in args and "b" in args


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_tool_result_loop(provider: str):
    """Full tool loop: user -> tool call -> tool result -> final response."""
    client, model = _make_client(provider)

    resp1 = await client.acompletion(
        model,
        [{"role": "user", "content": "What is 2+2? Use the add tool."}],
        tools=[ADD_TOOL],
    )
    assert resp1.choices[0].message.tool_calls is not None
    tc = resp1.choices[0].message.tool_calls[0]

    resp2 = await client.acompletion(
        model,
        [
            {"role": "user", "content": "What is 2+2? Use the add tool."},
            {
                "role": "assistant",
                "content": resp1.choices[0].message.content,
                "tool_calls": [tc.model_dump()],  # pyright: ignore[reportArgumentType]  # SDK expects serialized dicts
            },
            {"role": "tool", "tool_call_id": tc.id, "content": "4"},
        ],
        tools=[ADD_TOOL],
    )
    assert resp2.choices[0].finish_reason == "stop"
    assert resp2.choices[0].message.text


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_tool_result_loop_hardcoded(provider: str):
    """Single tool call roundtrip with hardcoded messages.

    Unlike test_tool_result_loop which builds messages from a live API response,
    this test controls the exact message shape to isolate provider conversion bugs
    (e.g. Google not converting role="tool" or using wrong function_response.name).
    """
    client, model = _make_client(provider)
    messages = [
        {"role": "user", "content": "What is 2+2? Use the add tool."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "4"},
    ]
    resp = await client.acompletion(model, messages, tools=[ADD_TOOL])  # pyright: ignore[reportArgumentType]
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.text


MULTIPLY_TOOL: ToolDefParam = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_tool_result_loop_parallel(provider: str):
    """Parallel tool calls -> 2 consecutive tool result messages -> final response.

    Reproduces the pattern from giskard-agents when a model requests multiple
    tools in a single turn. Each tool result is a separate role="tool" message.
    """
    client, model = _make_client(provider)
    messages = [
        {
            "role": "user",
            "content": "What is 2+2 and 3*4? Use the add and multiply tools.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
                },
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "arguments": '{"a": 3, "b": 4}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "4"},
        {"role": "tool", "tool_call_id": "call_1", "content": "12"},
    ]
    resp = await client.acompletion(model, messages, tools=[ADD_TOOL, MULTIPLY_TOOL])  # pyright: ignore[reportArgumentType]
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.text


# -- Usage scenarios -----------------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_usage_populated(provider: str):
    """Completion response includes usage with non-negative token counts."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model, [{"role": "user", "content": "Say one word."}]
    )
    assert resp.usage is not None
    assert resp.usage.input_tokens >= 0
    assert resp.usage.output_tokens >= 0
    assert resp.usage.total_tokens >= resp.usage.input_tokens


# -- Structured output scenarios -----------------------------------------------


class ColorModel(BaseModel):
    """All fields required (typical \"full\" structured-output shape)."""

    name: str
    hex: str


class JudgeLikeResult(BaseModel):
    """Shape aligned with checks that use optional ``reason`` (nullable, not required key)."""

    passed: bool
    reason: str | None = None


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_response_format(provider: str):
    """response_format with Pydantic model -> JSON that validates as the model (full required fields)."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model,
        [{"role": "user", "content": "Give me a color. Respond with name and hex."}],
        response_format=ColorModel,
    )
    choice = resp.choices[0]
    assert choice.message.text
    raw_json = choice.message.text
    assert isinstance(raw_json, str)

    parsed = json.loads(raw_json)
    validated = ColorModel.model_validate(parsed)
    assert isinstance(validated.name, str)
    assert isinstance(validated.hex, str)


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_response_format_optional_nullable_reason(provider: str):
    """Judge-like schema: ``passed`` plus optional nullable ``reason`` (matches LLM checks)."""
    client, model = _make_client(provider)
    resp = await client.acompletion(
        model,
        [
            {
                "role": "user",
                "content": (
                    "Judge whether 2 equals 2. Use the structured format. "
                    "Include a brief reason field when you explain."
                ),
            },
        ],
        response_format=JudgeLikeResult,
    )
    choice = resp.choices[0]
    assert choice.message.text
    raw_json = choice.message.text
    assert isinstance(raw_json, str)

    parsed = json.loads(raw_json)
    result = JudgeLikeResult.model_validate(parsed)
    assert isinstance(result.passed, bool)
    assert result.reason is None or isinstance(result.reason, str)


# -- LLMClient.configure() scenarios ------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_configure_explicit(provider: str):
    """Explicit configure() -> completion succeeds."""
    client, model = _make_client(provider)
    resp = await client.acompletion(model, [{"role": "user", "content": "Say hi"}])
    assert resp.choices[0].message.text


# -- Error handling scenarios --------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_invalid_api_key(provider: str):
    """Bogus API key -> AuthenticationError."""
    if provider == "bare":
        pytest.skip("bare path does not call configure(); no api_key to override")
    from giskard.llm.errors import AuthenticationError

    client = LLMClient()
    alias = f"bad-{provider}"
    bad_params = dict(_CONFIGURE_PARAMS[provider])
    bad_params["api_key"] = "invalid-key-12345"  # pragma: allowlist secret
    client.configure(alias, **bad_params)

    _, model_name = _MODELS[provider].split("/", 1)
    with pytest.raises(AuthenticationError):
        await client.acompletion(
            f"{alias}/{model_name}", [{"role": "user", "content": "Hi"}]
        )
