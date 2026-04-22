"""Tests for provider response conversion, error mapping, and validation."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.errors import BadRequestError, LLMError, RateLimitError
from giskard.llm.providers.anthropic import AnthropicProvider
from giskard.llm.providers.base import (
    CompletionProvider,
    EmbeddingProvider,
    ResponseProvider,
)
from giskard.llm.providers.google import GoogleProvider
from giskard.llm.providers.openai import OpenAIProvider
from giskard.llm.types import (
    ResponseOutputFunctionCall,
    ResponseOutputText,
    ToolCall,
)

# -- Helpers -------------------------------------------------------------------


def _make_openai_provider():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    return provider


def _make_google_provider():
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._client = MagicMock()
    return provider


def _make_anthropic_provider(merge_system: bool = False):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._merge_system = merge_system
    provider._client = MagicMock()
    return provider


def _make_openai_response(
    content: str | None = "Hello",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
):
    tc = None
    if tool_calls:
        tc = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tc,
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _make_openai_embedding_response(embeddings: list[list[float]]):
    return SimpleNamespace(
        data=[
            SimpleNamespace(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model="text-embedding-3-small",
        usage=SimpleNamespace(prompt_tokens=8, total_tokens=8),
    )


def _make_openai_response_api_response(
    id: str = "resp_001",
    output_items: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Mock the OpenAI Responses API shape."""
    if output_items is None:
        output_items = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello world")],
            )
        ]
    return SimpleNamespace(
        id=id,
        output=output_items,
        model="gpt-4o",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _make_google_interaction_response(
    id: str = "int_001",
    output_items: list[Any] | None = None,
    input_tokens: int = 8,
    output_tokens: int = 4,
):
    """Mock the Gemini Interactions API shape."""
    if output_items is None:
        output_items = [SimpleNamespace(type="text", text="Bonjour")]
    return SimpleNamespace(
        id=id,
        outputs=output_items,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


# -- OpenAI provider ----------------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello world")
    )

    resp = await provider.complete(
        "gpt-4o", [{"role": "user", "content": "Hi"}], temperature=0.5
    )
    assert resp.choices[0].message.content == "Hello world"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "gpt-4o"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 15


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion_with_typed_tool_calls(mock_import):
    mock_import.return_value = MagicMock()
    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
            },
        }
    ]
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(
            content=None, finish_reason="tool_calls", tool_calls=tool_calls
        )
    )

    resp = await provider.complete("gpt-4o", [{"role": "user", "content": "Weather?"}])
    assert resp.choices[0].finish_reason == "tool_calls"
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None
    assert isinstance(tcs[0], ToolCall)
    assert tcs[0].function.name == "get_weather"
    assert tcs[0].function.arguments == '{"city": "Paris"}'


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_embedding(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.embeddings = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
    provider._client.embeddings.create = AsyncMock(
        return_value=_make_openai_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    )

    resp = await provider.embed("text-embedding-3-small", ["hello", "world"])
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2]


# -- Error mapping completeness ------------------------------------------------


def _all_subclasses(cls: type) -> list[type]:
    """Recursively collect all subclasses of *cls*."""
    result: list[type] = []
    for sc in cls.__subclasses__():
        result.append(sc)
        result.extend(_all_subclasses(sc))
    return result


def _make_httpx_sdk_exc(cls: type) -> Exception:
    """Construct a minimal httpx-based SDK exception (openai / anthropic / google._interactions)."""
    import httpx  # pyright: ignore[reportMissingImports]

    name = cls.__name__
    if name == "APITimeoutError":
        return cls(request=httpx.Request("GET", "https://test"))  # type: ignore[call-arg]
    if name == "APIConnectionError":
        return cls(request=httpx.Request("GET", "https://test"), message="conn err")  # type: ignore[call-arg]
    if name == "APIResponseValidationError":
        resp = httpx.Response(200, request=httpx.Request("GET", "https://test"))
        return cls(response=resp, body=None)  # type: ignore[call-arg]
    # APIStatusError and all its subclasses
    resp = httpx.Response(400, request=httpx.Request("GET", "https://test"))
    return cls(message="test", response=resp, body=None)  # type: ignore[call-arg]


@pytest.mark.openai
def test_openai_map_error_completeness():
    """Every openai.APIError subclass must be mapped to an LLMError."""
    import openai  # pyright: ignore[reportMissingImports]

    provider = _make_openai_provider()
    for exc_cls in _all_subclasses(openai.APIError):
        with pytest.raises(LLMError):
            provider._map_error(_make_httpx_sdk_exc(exc_cls))


@pytest.mark.anthropic
def test_anthropic_map_error_completeness():
    """Every anthropic.APIError subclass must be mapped to an LLMError."""
    import anthropic  # pyright: ignore[reportMissingImports]

    provider = _make_anthropic_provider()
    for exc_cls in _all_subclasses(anthropic.APIError):
        with pytest.raises(LLMError):
            provider._map_error(_make_httpx_sdk_exc(exc_cls))


@pytest.mark.google
def test_google_map_error_completeness():
    """Every google.genai error must be mapped to an LLMError."""
    from google.genai import (  # pyright: ignore[reportMissingImports]
        errors as genai_errors,
    )

    provider = _make_google_provider()

    # google.genai.errors hierarchy (ClientError, ServerError)
    for exc_cls in [genai_errors.ClientError, genai_errors.ServerError]:
        with pytest.raises(LLMError):
            provider._map_error(exc_cls(400, "test"))

    # google.genai._interactions hierarchy (httpx-based, same shape as openai)
    try:
        from google.genai import (  # pyright: ignore[reportMissingImports]
            _interactions as ix,
        )

        for exc_cls in _all_subclasses(ix.APIError):
            with pytest.raises(LLMError):
                provider._map_error(_make_httpx_sdk_exc(exc_cls))
    except (ImportError, AttributeError):
        pass

    # Timeout heuristic (non-SDK exceptions with "timed out" in message)
    with pytest.raises(LLMError):
        provider._map_error(Exception("Connection timed out"))


# -- OpenAI message validation ------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_empty_messages(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="must not be empty"):
        await provider.complete("gpt-4o", [])


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_system_only(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="non-system message"):
        await provider.complete("gpt-4o", [{"role": "system", "content": "Be helpful"}])


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_empty_system_content(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="non-empty content"):
        await provider.complete(
            "gpt-4o",
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_multiple_system_works(mock_import):
    """OpenAI supports multiple system messages natively."""
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello")
    )
    resp = await provider.complete(
        "gpt-4o",
        [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello"


# -- Anthropic message validation ----------------------------------------------


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_multi_system_raises(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()

    with pytest.raises(BadRequestError, match="multiple system messages"):
        await provider.complete(
            "claude-3",
            [
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_multi_system_with_merge(mock_import):
    mock_anthropic = MagicMock()
    mock_import.return_value = mock_anthropic

    provider = _make_anthropic_provider(merge_system=True)

    mock_raw = MagicMock()
    mock_raw.content = [SimpleNamespace(type="text", text="Hello")]
    mock_raw.stop_reason = "end_turn"
    mock_raw.model = "claude-3"
    mock_raw.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    provider._client.messages.create = AsyncMock(return_value=mock_raw)

    resp = await provider.complete(
        "claude-3",
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello"


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_alternation(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()

    with pytest.raises(BadRequestError, match="alternating"):
        await provider.complete(
            "claude-3",
            [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "Hello again"},
            ],
        )


# -- OpenAI Responses API (respond) -------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_text(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.responses = MagicMock()
    provider._client.responses.create = AsyncMock(
        return_value=_make_openai_response_api_response()
    )

    resp = await provider.respond("gpt-4o", "Hello")
    assert resp.id == "resp_001"
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputText)
    assert resp.outputs[0].text == "Hello world"
    assert resp.output_text == "Hello world"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 10


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_function_call(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.responses = MagicMock()

    fc_item = SimpleNamespace(
        type="function_call",
        call_id="call_123",
        name="get_weather",
        arguments=json.dumps({"city": "Paris"}),
    )
    provider._client.responses.create = AsyncMock(
        return_value=_make_openai_response_api_response(output_items=[fc_item])
    )

    resp = await provider.respond("gpt-4o", "What's the weather?")
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputFunctionCall)
    assert resp.outputs[0].name == "get_weather"
    assert resp.outputs[0].arguments == {"city": "Paris"}
    assert resp.outputs[0].call_id == "call_123"


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_error_mapping(mock_import):
    openai = pytest.importorskip("openai")
    mock_import.return_value = openai

    provider = _make_openai_provider()
    provider._client.responses = MagicMock()

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "rate limited"}}
    err = openai.RateLimitError(
        message="rate limited",
        response=mock_response,
        body={"error": {"message": "rate limited"}},
    )
    provider._client.responses.create = AsyncMock(side_effect=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.respond("gpt-4o", "Hello")
    assert exc_info.value.status_code == 429


# -- Google Interactions API (respond) -----------------------------------------


@patch("giskard.llm.providers.google._import_genai_errors")
async def test_google_respond_text(mock_errors):
    mock_errors.return_value = MagicMock()
    provider = _make_google_provider()
    provider._client.aio = MagicMock()
    provider._client.aio.interactions = MagicMock()
    provider._client.aio.interactions.create = AsyncMock(
        return_value=_make_google_interaction_response()
    )

    resp = await provider.respond("gemini-2.0-flash", "Hello")
    assert resp.id == "int_001"
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputText)
    assert resp.outputs[0].text == "Bonjour"


@patch("giskard.llm.providers.google._import_genai_errors")
async def test_google_respond_function_call(mock_errors):
    mock_errors.return_value = MagicMock()
    provider = _make_google_provider()
    provider._client.aio = MagicMock()
    provider._client.aio.interactions = MagicMock()

    fc_item = SimpleNamespace(
        type="function_call",
        id="call_xyz",
        name="get_weather",
        arguments={"city": "Tokyo"},
    )
    provider._client.aio.interactions.create = AsyncMock(
        return_value=_make_google_interaction_response(output_items=[fc_item])
    )

    resp = await provider.respond("gemini-2.0-flash", "Weather?")
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputFunctionCall)
    assert resp.outputs[0].call_id == "call_xyz"
    assert resp.outputs[0].name == "get_weather"
    assert resp.outputs[0].arguments == {"city": "Tokyo"}


# -- Protocol conformance checks -----------------------------------------------


def test_openai_implements_all_protocols():
    provider = _make_openai_provider()
    assert isinstance(provider, CompletionProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, ResponseProvider)


def test_anthropic_implements_completion_only():
    provider = _make_anthropic_provider()
    assert isinstance(provider, CompletionProvider)
    assert not isinstance(provider, EmbeddingProvider)
    assert not isinstance(provider, ResponseProvider)


def test_google_implements_all_protocols():
    provider = _make_google_provider()
    assert isinstance(provider, CompletionProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, ResponseProvider)


# -- Google _convert_messages --------------------------------------------------


class TestGoogleConvertMessages:
    """Unit tests for GoogleProvider._convert_messages."""

    def _convert(self, messages):
        provider = _make_google_provider()
        return provider._convert_messages(messages)

    def test_user_message(self):
        result = self._convert([{"role": "user", "content": "Hello"}])
        assert result == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_assistant_becomes_model(self):
        result = self._convert([{"role": "assistant", "content": "Hi"}])
        assert result == [{"role": "model", "parts": [{"text": "Hi"}]}]

    def test_system_messages_skipped(self):
        result = self._convert(
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_role_produces_function_response(self):
        """tool messages should produce a function_response part."""
        result = self._convert(
            [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]
        )
        assert len(result) == 1
        fr = result[0]["parts"][0]["function_response"]
        assert fr["response"] == {"result": "42"}

    def test_tool_role_uses_function_name_from_preceding_call(self):
        """tool_call_id should resolve to the actual function name from the
        preceding assistant tool_calls, not be used verbatim as the name."""
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
            ]
        )
        fr = result[1]["parts"][0]["function_response"]
        assert fr["name"] == "get_weather"

    def test_tool_role_converted_to_user(self):
        """Google API expects role='user' for function_response parts."""
        result = self._convert(
            [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]
        )
        assert result[0]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 1, "b": 2}',
                            },
                        }
                    ],
                }
            ]
        )
        assert result[0]["role"] == "model"
        fc = result[0]["parts"][0]["function_call"]
        assert fc["name"] == "add"
        assert fc["args"] == {"a": 1, "b": 2}

    def test_multiple_tool_calls_in_one_message(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "multiply", "arguments": "{}"},
                        },
                    ],
                }
            ]
        )
        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0]["function_call"]["name"] == "add"
        assert result[0]["parts"][1]["function_call"]["name"] == "multiply"


# -- Google _normalize_input_items (Interactions API) --------------------------


class TestGoogleNormalizeInputItems:
    """Unit tests for GoogleProvider._normalize_input_items."""

    def _normalize(self, items):
        provider = _make_google_provider()
        return provider._normalize_input_items(items)

    def test_function_call_output_to_function_result(self):
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": {},
            },
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]
        result = self._normalize(items)
        assert result[1] == {
            "type": "function_result",
            "call_id": "c1",
            "name": "get_weather",
            "result": "sunny",
        }

    def test_function_call_output_explicit_name_wins(self):
        """If the item already carries a name, it takes precedence over the map."""
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": {},
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "name": "override",
                "output": "x",
            },
        ]
        result = self._normalize(items)
        assert result[1]["name"] == "override"

    def test_function_call_output_unknown_call_id(self):
        items = [
            {"type": "function_call_output", "call_id": "unknown", "output": "x"},
        ]
        result = self._normalize(items)
        assert result[0]["name"] == ""
        assert result[0]["call_id"] == "unknown"

    def test_function_call_renames_call_id_to_id(self):
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "fn",
                "arguments": {"a": 1},
            },
        ]
        result = self._normalize(items)
        assert result[0]["id"] == "c1"
        assert "call_id" not in result[0]

    def test_function_call_keeps_id_field(self):
        """Items using 'id' instead of 'call_id' should pass through correctly."""
        items = [
            {"type": "function_call", "id": "c2", "name": "fn", "arguments": {}},
        ]
        result = self._normalize(items)
        assert result[0]["id"] == "c2"

    def test_function_call_arguments_dict_kept(self):
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "fn",
                "arguments": {"k": "v"},
            },
        ]
        result = self._normalize(items)
        assert result[0]["arguments"] == {"k": "v"}
        assert isinstance(result[0]["arguments"], dict)

    def test_function_call_arguments_json_string_parsed(self):
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "fn",
                "arguments": '{"k": "v"}',
            },
        ]
        result = self._normalize(items)
        assert result[0]["arguments"] == {"k": "v"}
        assert isinstance(result[0]["arguments"], dict)

    def test_role_tagged_turn_to_text_content_param(self):
        items = [{"role": "user", "content": "Hello"}]
        result = self._normalize(items)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_role_tagged_turn_empty_content(self):
        items = [{"role": "user", "content": ""}]
        result = self._normalize(items)
        assert result == [{"type": "text", "text": ""}]

    def test_role_tagged_turn_none_content(self):
        items = [{"role": "user", "content": None}]
        result = self._normalize(items)
        assert result == [{"type": "text", "text": ""}]

    def test_passthrough_valid_text_content_param(self):
        items = [{"type": "text", "text": "already valid"}]
        result = self._normalize(items)
        assert result == [{"type": "text", "text": "already valid"}]

    def test_passthrough_unknown_type(self):
        items = [{"type": "something_else", "data": 1}]
        result = self._normalize(items)
        assert result == [{"type": "something_else", "data": 1}]

    def test_mixed_list_full_roundtrip(self):
        """A realistic input: user text + function_call echo + function_call_output."""
        items = [
            {"role": "user", "content": "What's the weather?"},
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
            {"type": "function_call_output", "call_id": "fc_1", "output": "rainy"},
        ]
        result = self._normalize(items)
        assert result[0] == {"type": "text", "text": "What's the weather?"}
        assert result[1] == {
            "type": "function_call",
            "id": "fc_1",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
        assert result[2] == {
            "type": "function_result",
            "call_id": "fc_1",
            "name": "get_weather",
            "result": "rainy",
        }


# -- Anthropic _convert_messages -----------------------------------------------


class TestAnthropicConvertMessages:
    """Unit tests for AnthropicProvider._convert_messages."""

    def _convert(self, messages) -> list[dict[str, Any]]:
        provider = _make_anthropic_provider()
        return provider._convert_messages(messages)  # pyright: ignore[reportReturnType]

    def test_user_message(self):
        result = self._convert([{"role": "user", "content": "Hello"}])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        result = self._convert([{"role": "assistant", "content": "Hi"}])
        assert result == [{"role": "assistant", "content": "Hi"}]

    def test_tool_becomes_user_with_tool_result(self):
        result = self._convert(
            [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_1"
        assert block["content"] == "42"

    def test_consecutive_tool_messages_merged(self):
        """Consecutive tool messages must be merged into a single user message
        with multiple tool_result blocks, to satisfy Anthropic's alternation."""
        result = self._convert(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "a"},
                {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[0]["content"][1]["tool_use_id"] == "call_2"

    def test_assistant_with_tool_calls(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 1, "b": 2}',
                            },
                        }
                    ],
                }
            ]
        )
        assert result[0]["role"] == "assistant"
        content = result[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "add"
        assert content[0]["input"] == {"a": 1, "b": 2}

    def test_assistant_with_content_and_tool_calls(self):
        result = self._convert(
            [
                {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "test"}',
                            },
                        }
                    ],
                }
            ]
        )
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me check"
        assert content[1]["type"] == "tool_use"

    def test_full_tool_roundtrip_sequence(self):
        """user -> assistant(tool_calls) -> tool -> should produce valid
        alternating user/assistant/user sequence."""
        result = self._convert(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":2,"b":2}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            ]
        )
        assert len(result) == 3
        assert [m["role"] for m in result] == ["user", "assistant", "user"]


# -- AzureAIProvider base_url shaping -------------------------------------------


class TestAzureAIFoundryBaseUrlShaping:
    """``AzureAIProvider`` auto-suffixes bare Foundry root URLs with ``/models``.

    Matches litellm's ``azure_ai/`` URL shaping so existing
    ``AZURE_AI_API_BASE`` values (the bare Foundry root) keep working.
    """

    @pytest.mark.parametrize(
        ("input_url", "expected"),
        [
            pytest.param(
                "https://dev.services.ai.azure.com",
                "https://dev.services.ai.azure.com/models",
                id="foundry-root-no-slash",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/",
                "https://dev.services.ai.azure.com/models",
                id="foundry-root-trailing-slash",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/models",
                "https://dev.services.ai.azure.com/models",
                id="foundry-with-models-path-unchanged",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/openai/v1",
                "https://dev.services.ai.azure.com/openai/v1",
                id="foundry-with-openai-v1-path-unchanged",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com?api-version=2024-05-01-preview",
                "https://dev.services.ai.azure.com/models?api-version=2024-05-01-preview",
                id="foundry-root-preserves-query",
            ),
            pytest.param(
                "https://custom.example.com",
                "https://custom.example.com",
                id="non-foundry-host-unchanged",
            ),
            pytest.param(
                "https://my-resource.openai.azure.com",
                "https://my-resource.openai.azure.com",
                id="classic-azure-openai-host-unchanged",
            ),
            pytest.param(None, None, id="none-passthrough"),
            pytest.param("", "", id="empty-passthrough"),
        ],
    )
    def test_shape_foundry_base_url(self, input_url, expected):
        from giskard.llm.providers.azure_ai import _shape_foundry_base_url

        assert _shape_foundry_base_url(input_url) == expected

    def test_provider_init_shapes_env_endpoint(self, monkeypatch):
        """``AzureAIProvider()`` shapes the URL read from ``AZURE_AI_ENDPOINT``
        before handing it to ``openai.AsyncOpenAI``."""
        from giskard.llm.providers.azure_ai import AzureAIProvider

        monkeypatch.setenv("AZURE_AI_API_KEY", "k")
        monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://dev.services.ai.azure.com")

        with patch("openai.AsyncOpenAI") as mock_client:
            AzureAIProvider()

        kwargs = mock_client.call_args.kwargs
        assert kwargs["base_url"] == "https://dev.services.ai.azure.com/models"
        assert kwargs["api_key"] == "k"

    def test_provider_init_preserves_explicit_path(self, monkeypatch):
        """Explicit sub-path in ``base_url`` kwarg is preserved unchanged."""
        from giskard.llm.providers.azure_ai import AzureAIProvider

        monkeypatch.delenv("AZURE_AI_ENDPOINT", raising=False)

        with patch("openai.AsyncOpenAI") as mock_client:
            AzureAIProvider(
                api_key="k",
                base_url="https://dev.services.ai.azure.com/openai/v1",
            )

        assert (
            mock_client.call_args.kwargs["base_url"]
            == "https://dev.services.ai.azure.com/openai/v1"
        )
