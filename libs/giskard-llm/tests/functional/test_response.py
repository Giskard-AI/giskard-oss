"""Functional tests for ResponseProvider (Responses/Interactions API).

Only OpenAI and Google support these APIs. Each test is parametrized by provider
and asserts on structure (non-empty id, correct output types), not content.
"""

import os

import pytest
from giskard.llm import LLMClient
from giskard.llm.errors import AuthenticationError, LLMError, UnsupportedOperationError
from giskard.llm.types import ResponseFunctionToolCall, ToolDefParam

pytestmark = pytest.mark.functional

# -- Provider parametrization -------------------------------------------------

_RESPONSE_MODELS = {
    "openai": os.getenv("TEST_OPENAI_MODEL", "openai/gpt-4.1-nano"),
    "google": os.getenv("TEST_GOOGLE_MODEL", "google/gemini-2.5-flash"),
}

_CONFIGURE_PARAMS = {  # pragma: allowlist secret
    "openai": {"provider": "openai", "api_key": "os.environ/OPENAI_API_KEY"},
    "google": {"provider": "google", "api_key": "os.environ/GOOGLE_API_KEY"},
}

_PROVIDER_MARKS = {
    "openai": pytest.mark.openai,
    "google": pytest.mark.google,
}

_PROVIDER_PARAMS = [
    pytest.param(provider, marks=_PROVIDER_MARKS[provider], id=provider)
    for provider in _RESPONSE_MODELS
]


def _make_client(provider: str) -> tuple[LLMClient, str]:
    """Create a configured LLMClient and return (client, model_string)."""
    client = LLMClient()
    alias = f"test-{provider}"
    client.configure(alias, **_CONFIGURE_PARAMS[provider])
    model = _RESPONSE_MODELS[provider]
    _, model_name = model.split("/", 1)
    return client, f"{alias}/{model_name}"


# -- Basic response scenarios -------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_text_input(provider: str):
    """String input -> non-empty text output with a valid id."""
    client, model = _make_client(provider)
    resp = await client.aresponse(model, "Say hello")
    assert resp.id
    assert len(resp.outputs) > 0
    assert resp.output_text is not None
    assert len(resp.output_text.strip()) > 0


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_with_instructions(provider: str):
    """Instructions param -> keyword appears in output."""
    client, model = _make_client(provider)
    resp = await client.aresponse(
        model,
        "Tell me something.",
        instructions="Always include the word PINEAPPLE in your response.",
    )
    assert resp.output_text is not None
    assert "pineapple" in resp.output_text.lower()


# -- Tool call scenario -------------------------------------------------------

_ADD_TOOL: ToolDefParam = {
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
async def test_respond_function_call(provider: str):
    """Tools param with a function-triggering prompt -> function call output."""
    client, model = _make_client(provider)
    resp = await client.aresponse(
        model,
        "What is 2+2? Use the add tool.",
        tools=[_ADD_TOOL],
    )
    fc_outputs = [o for o in resp.outputs if isinstance(o, ResponseFunctionToolCall)]
    assert len(fc_outputs) > 0
    fc = fc_outputs[0]
    assert fc.name == "add"
    assert "a" in fc.arguments and "b" in fc.arguments


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_tool_roundtrip(provider: str):
    """Tool roundtrip via input list: tool call -> feed back result -> final text.

    Unlike test_respond_stateful_with_tool_result which uses previous_id,
    this test feeds the full conversation (including tool results) as input.
    """
    if provider == "google":
        return pytest.skip("Google returns 'Request contains an invalid argument.'")

    client, model = _make_client(provider)
    resp1 = await client.aresponse(
        model, "What is 2+2? Use the add tool.", tools=[_ADD_TOOL]
    )
    fc_outputs = [o for o in resp1.outputs if isinstance(o, ResponseFunctionToolCall)]
    assert len(fc_outputs) > 0
    fc = fc_outputs[0]

    resp2 = await client.aresponse(
        model,
        [
            {"role": "user", "content": "What is 2+2? Use the add tool."},
            {
                "type": "function_call",
                "name": fc.name,
                "call_id": fc.call_id or fc.name,
                "arguments": fc.arguments,
            },
            {
                "type": "function_call_output",
                "call_id": fc.call_id or fc.name,
                "name": fc.name,
                "output": "4",
            },
        ],
        tools=[_ADD_TOOL],
    )
    assert resp2.output_text is not None
    assert resp2.output_text.strip()


# -- Stateful turn scenario ---------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_stateful_turn(provider: str):
    """Two calls with previous_id -> model remembers context."""
    client, model = _make_client(provider)

    resp1 = await client.aresponse(
        model,
        "My name is Zephyr. Remember that.",
        instructions="You are a helpful assistant with good memory.",
    )
    assert resp1.id

    resp2 = await client.aresponse(
        model,
        "What is my name?",
        previous_id=resp1.id,
    )
    assert resp2.output_text is not None
    assert "zephyr" in resp2.output_text.lower()


# -- Multi-turn list[dict] input scenario -------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_multi_turn_input(provider: str):
    """List-of-dicts input (multi-turn conversation) -> valid response."""
    client, model = _make_client(provider)
    resp = await client.aresponse(
        model,
        [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Got it, blue!"},
            {"role": "user", "content": "What is my favorite color?"},
        ],
    )
    assert resp.id
    assert resp.output_text is not None
    assert "blue" in resp.output_text.lower()


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_chained_multi_turn(provider: str):
    """Multiple list[dict] calls chained via previous_id."""
    client, model = _make_client(provider)

    resp1 = await client.aresponse(
        model,
        [{"role": "user", "content": "Remember: the secret word is MANGO."}],
        instructions="You are a helpful assistant with perfect memory.",
    )
    assert resp1.id

    resp2 = await client.aresponse(
        model,
        [{"role": "user", "content": "Now remember: my lucky number is 42."}],
        previous_id=resp1.id,
    )
    assert resp2.id

    resp3 = await client.aresponse(
        model,
        [{"role": "user", "content": "What is the secret word and my lucky number?"}],
        previous_id=resp2.id,
    )
    assert resp3.output_text is not None
    text = resp3.output_text.lower()
    assert "mango" in text
    assert "42" in text


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_stateful_with_tool_result(provider: str):
    """Stateful turn with list[dict] input feeding back a tool result."""
    client, model = _make_client(provider)

    resp1 = await client.aresponse(
        model,
        "What is 3+4? Use the add tool.",
        tools=[_ADD_TOOL],
    )
    assert resp1.id
    fc_outputs = resp1.function_calls
    assert len(fc_outputs) > 0

    fc = fc_outputs[0]
    resp2 = await client.aresponse(
        model,
        [
            {
                "type": "function_call_output",
                "call_id": fc.call_id or "call_0",
                "name": fc.name,
                "output": "7",
            },
        ],
        previous_id=resp1.id,
    )
    assert resp2.id
    assert resp2.output_text is not None
    assert "7" in resp2.output_text


# -- Usage fields scenario ----------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_usage_populated(provider: str):
    """Response includes usage with non-negative token counts."""
    client, model = _make_client(provider)
    resp = await client.aresponse(model, "Say one word.")
    assert resp.usage is not None
    assert resp.usage.input_tokens >= 0
    assert resp.usage.output_tokens >= 0
    assert resp.usage.total_tokens >= resp.usage.input_tokens


# -- Error path scenario ------------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_invalid_model_raises(provider: str):
    """Non-existent model name -> LLMError from _map_error."""
    client, model = _make_client(provider)
    alias = model.split("/", 1)[0]
    with pytest.raises(LLMError):
        await client.aresponse(f"{alias}/nonexistent-model-xyz-999", "Hello")


# -- Unsupported provider scenario --------------------------------------------


@pytest.mark.anthropic
async def test_respond_unsupported_provider():
    """Anthropic provider -> clear error when calling arespond()."""
    client = LLMClient()
    client.configure(
        "test-anthropic",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    with pytest.raises(UnsupportedOperationError, match="does not support"):
        await client.aresponse("test-anthropic/claude-haiku-4-5-20251001", "Hello")


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_invalid_api_key(provider: str):
    """Bogus API key -> AuthenticationError on Response API."""
    client = LLMClient()
    alias = f"bad-{provider}"
    bad_params = dict(_CONFIGURE_PARAMS[provider])
    bad_params["api_key"] = "invalid-key-12345"  # pragma: allowlist secret
    client.configure(alias, **bad_params)

    _, model_name = _RESPONSE_MODELS[provider].split("/", 1)
    with pytest.raises(AuthenticationError):
        await client.aresponse(f"{alias}/{model_name}", "Hello")


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_empty_input(provider: str):
    """Empty list input -> LLMError or BadRequestError."""
    client, model = _make_client(provider)
    with pytest.raises(LLMError):
        await client.aresponse(model, [])
