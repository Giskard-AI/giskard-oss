"""Tests for provider response conversion, error mapping, and validation."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.errors import BadRequestError, LLMError, RateLimitError
from giskard.llm.providers.anthropic import AnthropicProvider
from giskard.llm.providers.azure_ai import AzureAIProvider
from giskard.llm.providers.azure_openai import AzureOpenAIProvider
from giskard.llm.providers.base import (
    CompletionProvider,
    EmbeddingProvider,
    ResponseProvider,
)
from giskard.llm.providers.google import GoogleProvider
from giskard.llm.providers.openai import OpenAIProvider
from giskard.llm.types import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    TextContent,
    ToolCall,
)

if TYPE_CHECKING:
    from google.genai.types import HttpOptionsOrDict
    from httpx import AsyncClient

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
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ] = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
):
    """Build a :class:`openai.types.chat.chat_completion.ChatCompletion` for mocks."""
    pytest.importorskip("openai")
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCallUnion,
    )
    from openai.types.completion_usage import CompletionUsage

    tc_list: list[ChatCompletionMessageToolCallUnion] | None = None
    if tool_calls:
        tc_list = cast(
            list[ChatCompletionMessageToolCallUnion],
            [
                ChatCompletionMessageFunctionToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ],
        )
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tc_list,
                ),
            )
        ],
        created=0,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(
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
                role="assistant",
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
    steps: list[Any] | None = None,
    input_tokens: int = 8,
    output_tokens: int = 4,
):
    """Mock the Gemini Interactions API shape."""
    if steps is None:
        steps = [
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text="Bonjour")],
            )
        ]
    return SimpleNamespace(
        id=id,
        steps=steps,
        usage=SimpleNamespace(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


# -- OpenAI provider ----------------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
@pytest.mark.openai
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
@pytest.mark.openai
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
    assert tcs[0].function.arguments == {"city": "Paris"}


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


# -- Provider transport configuration -----------------------------------------


def test_openai_provider_forwards_transport_config():
    http_client = object()
    default_headers = {"x-test": "1"}

    with patch("giskard.llm.providers.openai._import_openai") as mock_import:
        sdk = MagicMock()
        mock_import.return_value = sdk

        OpenAIProvider(
            api_key="k",
            base_url="https://example.test/v1",
            timeout=12,
            http_client=cast("AsyncClient", http_client),
            default_headers=default_headers,
        )

    kwargs = sdk.AsyncOpenAI.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["base_url"] == "https://example.test/v1"
    assert kwargs["timeout"] == 12
    assert kwargs["http_client"] is http_client
    assert kwargs["default_headers"] == default_headers


def test_openai_provider_supports_azure_foundry_v1_base_url():
    with patch("giskard.llm.providers.openai._import_openai") as mock_import:
        sdk = MagicMock()
        mock_import.return_value = sdk

        OpenAIProvider(
            api_key="k",
            base_url="https://example.openai.azure.com/openai/v1/",
        )

    kwargs = sdk.AsyncOpenAI.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["base_url"] == "https://example.openai.azure.com/openai/v1/"
    assert "api_version" not in kwargs


def test_azure_openai_provider_forwards_transport_config(monkeypatch):
    http_client = object()
    default_headers = {"x-test": "1"}
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)

    with patch("openai.AsyncAzureOpenAI") as mock_client:
        AzureOpenAIProvider(
            api_key="k",
            base_url="https://azure.test",
            api_version="2024-10-21",
            timeout=12,
            http_client=cast("AsyncClient", http_client),
            default_headers=default_headers,
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["azure_endpoint"] == "https://azure.test"
    assert kwargs["api_version"] == "2024-10-21"
    assert kwargs["timeout"] == 12
    assert kwargs["http_client"] is http_client
    assert kwargs["default_headers"] == default_headers


def test_azure_ai_provider_forwards_transport_config(monkeypatch):
    http_client = object()
    default_headers = {"x-ms-useragent": "giskard-llm"}
    monkeypatch.delenv("AZURE_AI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_AI_API_VERSION", raising=False)

    with patch("openai.AsyncAzureOpenAI") as mock_client:
        AzureAIProvider(
            api_key="k",
            base_url="https://dev.services.ai.azure.com/models",
            timeout=12,
            http_client=cast("AsyncClient", http_client),
            default_headers=default_headers,
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["azure_endpoint"] == "https://dev.services.ai.azure.com"
    assert kwargs["api_version"] == "2024-10-21"
    assert kwargs["timeout"] == 12
    assert kwargs["http_client"] is http_client
    assert kwargs["default_headers"] == default_headers


def test_anthropic_provider_forwards_transport_config():
    http_client = object()
    default_headers = {"x-test": "1"}

    with patch("giskard.llm.providers.anthropic._import_anthropic") as mock_import:
        sdk = MagicMock()
        mock_import.return_value = sdk

        AnthropicProvider(
            api_key="k",
            base_url="https://anthropic.test",
            timeout=12,
            http_client=cast("AsyncClient", http_client),
            default_headers=default_headers,
        )

    kwargs = sdk.AsyncAnthropic.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["base_url"] == "https://anthropic.test"
    assert kwargs["timeout"] == 12
    assert kwargs["http_client"] is http_client
    assert kwargs["default_headers"] == default_headers


class _FakeHttpOptions:
    def __init__(self, httpxAsyncClient=None, headers=None):
        self.httpx_async_client = httpxAsyncClient
        self.headers = headers

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(f"Cannot validate {type(value)}")

    def model_copy(self, update=None):
        copied = type(self)(
            httpxAsyncClient=self.httpx_async_client,
            headers=self.headers,
        )
        for key, value in (update or {}).items():
            setattr(copied, key, value)
        return copied


def _patch_google_sdk(monkeypatch):
    sdk = MagicMock()
    types = MagicMock(HttpOptions=_FakeHttpOptions)
    monkeypatch.setattr("giskard.llm.providers.google._import_genai", lambda: sdk)
    monkeypatch.setattr(
        "giskard.llm.providers.google._import_genai_types", lambda: types
    )
    return sdk


def test_google_provider_builds_http_options_from_transport_config(monkeypatch):
    http_client = object()
    default_headers = {"x-test": "1"}
    sdk = _patch_google_sdk(monkeypatch)

    GoogleProvider(
        api_key="k",
        http_client=cast("AsyncClient", http_client),
        default_headers=default_headers,
    )

    kwargs = sdk.Client.call_args.kwargs
    assert kwargs["api_key"] == "k"
    assert kwargs["http_options"].httpx_async_client is http_client
    assert kwargs["http_options"].headers == default_headers


def test_google_provider_no_transport_kwargs_passes_no_http_options(monkeypatch):
    sdk = _patch_google_sdk(monkeypatch)

    GoogleProvider(api_key="k")

    kwargs = sdk.Client.call_args.kwargs
    assert "http_options" not in kwargs or kwargs["http_options"] is None


def test_google_provider_preserves_explicit_http_options_and_fills_missing(monkeypatch):
    explicit_client = object()
    fallback_client = object()
    fallback_headers = {"x-fallback": "1"}
    sdk = _patch_google_sdk(monkeypatch)

    GoogleProvider(
        api_key="k",
        http_client=cast("AsyncClient", fallback_client),
        default_headers=fallback_headers,
        http_options=_FakeHttpOptions(httpxAsyncClient=explicit_client),  # pyright: ignore[reportArgumentType]
    )

    options = sdk.Client.call_args.kwargs["http_options"]
    assert options.httpx_async_client is explicit_client
    assert options.headers == fallback_headers


def test_google_provider_warns_when_http_options_conflicts_with_convenience_kwargs(
    monkeypatch, caplog
):
    explicit_client = object()
    explicit_headers = {"x-explicit": "1"}
    conflicting_client = object()
    conflicting_headers = {"x-conflict": "1"}
    sdk = _patch_google_sdk(monkeypatch)

    import logging

    with caplog.at_level(logging.WARNING, logger="giskard.llm.providers.google"):
        GoogleProvider(
            api_key="k",
            http_client=cast("AsyncClient", conflicting_client),
            default_headers=conflicting_headers,
            http_options=_FakeHttpOptions(
                httpxAsyncClient=explicit_client, headers=explicit_headers
            ),  # pyright: ignore[reportArgumentType]
        )

    options = sdk.Client.call_args.kwargs["http_options"]
    assert options.httpx_async_client is explicit_client
    assert options.headers == explicit_headers
    assert any("http_client" in r.message for r in caplog.records)
    assert any("default_headers" in r.message for r in caplog.records)


def test_google_provider_raises_for_invalid_http_options(monkeypatch):
    _patch_google_sdk(monkeypatch)

    with pytest.raises(ValueError, match="invalid http_options"):
        GoogleProvider(api_key="k", http_options=cast("HttpOptionsOrDict", object()))


# -- Error mapping completeness ------------------------------------------------


def _all_subclasses(cls: type) -> list[type]:
    """Recursively collect all subclasses of *cls*."""
    result: list[type] = []
    for sc in cls.__subclasses__():
        result.append(sc)
        result.extend(_all_subclasses(sc))
    return result


def _make_httpx_sdk_exc(cls: type) -> Exception:
    """Construct a minimal httpx-based SDK exception (openai / anthropic / google._interactions).

    SDK error classes have inconsistent constructor signatures (some take
    ``message``, some ``response``, some ``request``, some only keyword args).
    Rather than special-casing each class name, introspect the constructor and
    pass only the arguments it accepts.
    """
    import inspect

    import httpx

    request = httpx.Request("GET", "https://test")
    response = httpx.Response(400, request=request)

    available = {
        "message": "test",
        "request": request,
        "response": response,
        "body": None,
    }

    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, object] = {
        name: value for name, value in available.items() if name in params
    }
    return cls(**kwargs)


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
@pytest.mark.openai
async def test_openai_validate_developer_content_as_content_blocks(mock_import):
    """Developer ``content`` typed as a list of ``TextContent`` blocks (the SDK-native
    shape, reachable via plain-dict validation) must be normalized through ``.text``,
    not crash with ``AttributeError`` from calling ``.strip()`` on a list."""
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello")
    )
    resp = await provider.complete(
        "gpt-4o",
        [
            {"role": "developer", "content": [{"type": "text", "text": "Be helpful"}]},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello"


@patch("giskard.llm.providers.openai._import_openai")
@pytest.mark.openai
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
    assert resp.choices[0].message.content == [TextContent(text="Hello")]


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_consecutive_developers_ok(mock_import):
    """Developer turns are folded into ``system``; multiple developer messages require merge_system=True."""
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
            {"role": "developer", "content": "First instruction."},
            {"role": "developer", "content": "Second instruction."},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == [TextContent(text="Hello")]
    assert resp.choices[0].message.tool_calls is None


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_system_and_developer_raises_without_merge(
    mock_import,
):
    """A system + developer combo counts as two instruction messages and requires merge_system=True."""
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()
    with pytest.raises(BadRequestError, match="multiple system messages"):
        await provider.complete(
            "claude-3",
            [
                {"role": "system", "content": "System instruction."},
                {"role": "developer", "content": "Developer instruction."},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_empty_developer_content(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()
    with pytest.raises(BadRequestError, match="non-empty content"):
        await provider.complete(
            "claude-3",
            [
                {"role": "developer", "content": ""},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_developer_content_as_content_blocks(mock_import):
    """Developer ``content`` typed as a list of ``TextContent`` blocks must be
    normalized through ``.text``, not crash with ``AttributeError`` from calling
    ``.strip()`` on a list."""
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()

    mock_raw = MagicMock()
    mock_raw.content = [SimpleNamespace(type="text", text="Hello")]
    mock_raw.stop_reason = "end_turn"
    mock_raw.model = "claude-3"
    mock_raw.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    provider._client.messages.create = AsyncMock(return_value=mock_raw)

    resp = await provider.complete(
        "claude-3",
        [
            {"role": "developer", "content": [{"type": "text", "text": "Be helpful"}]},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == [TextContent(text="Hello")]


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


# -- Google message validation --------------------------------------------------


@patch("giskard.llm.providers.google._import_genai_errors")
@pytest.mark.google
async def test_google_validate_developer_content_as_content_blocks(mock_errors):
    """Developer ``content`` typed as a list of ``TextContent`` blocks must be
    normalized through ``.text``, not crash with ``AttributeError`` from calling
    ``.strip()`` on a list."""
    genai_types = pytest.importorskip("google.genai.types")
    mock_errors.return_value = MagicMock()
    provider = _make_google_provider()
    provider._client.aio = MagicMock()
    provider._client.aio.models = MagicMock()
    raw = genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 3,
                "candidates_token_count": 4,
                "total_token_count": 7,
            },
        }
    )
    provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

    resp = await provider.complete(
        "gemini-3.5-flash",
        [
            {"role": "developer", "content": [{"type": "text", "text": "Be helpful"}]},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == [TextContent(text="Hello")]


# -- OpenAI Responses API (respond) -------------------------------------------


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

    resp = await provider.respond("gemini-3.5-flash", "Hello")
    assert resp.id == "int_001"
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputMessage)
    assert isinstance(resp.outputs[0].content[0], ResponseOutputText)
    assert resp.outputs[0].content[0].text == "Bonjour"


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
        return_value=_make_google_interaction_response(steps=[fc_item])
    )

    resp = await provider.respond("gemini-3.5-flash", "Weather?")
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseFunctionToolCall)
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


# -- AzureAIProvider Foundry endpoint normalization -----------------------------


class TestAzureAiEndpointNormalization:
    """Normalize ``*.services.ai.azure.com`` roots for ``AsyncAzureOpenAI``."""

    @pytest.mark.parametrize(
        ("input_url", "expected"),
        [
            pytest.param(
                "https://dev.services.ai.azure.com",
                "https://dev.services.ai.azure.com",
                id="foundry-root-no-slash",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/",
                "https://dev.services.ai.azure.com",
                id="foundry-root-trailing-slash",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/models",
                "https://dev.services.ai.azure.com",
                id="foundry-models-suffix-stripped",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/models/",
                "https://dev.services.ai.azure.com",
                id="foundry-models-trailing-slash-stripped",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com/openai/v1",
                "https://dev.services.ai.azure.com/openai/v1",
                id="foundry-with-openai-v1-path-unchanged",
            ),
            pytest.param(
                "https://dev.services.ai.azure.com?api-version=2024-05-01-preview",
                "https://dev.services.ai.azure.com",
                id="foundry-root-query-dropped-normalized-root",
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
    def test_normalize_azure_ai_endpoint(self, input_url, expected):
        from giskard.llm.providers.azure_ai import _normalize_azure_ai_endpoint

        assert _normalize_azure_ai_endpoint(input_url) == expected

    def test_provider_init_uses_async_azure_openai(self, monkeypatch):
        """``AzureAIProvider()`` passes a normalized Foundry root to ``AsyncAzureOpenAI``."""
        from giskard.llm.providers.azure_ai import AzureAIProvider

        monkeypatch.setenv("AZURE_AI_API_KEY", "k")
        monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://dev.services.ai.azure.com")
        monkeypatch.delenv("AZURE_AI_API_VERSION", raising=False)

        with patch("openai.AsyncAzureOpenAI") as mock_client:
            AzureAIProvider()

        kwargs = mock_client.call_args.kwargs
        assert kwargs["azure_endpoint"] == "https://dev.services.ai.azure.com"
        assert kwargs["api_key"] == "k"
        assert kwargs["api_version"] == "2024-10-21"

    def test_provider_init_preserves_explicit_path(self, monkeypatch):
        """Explicit sub-path in ``base_url`` kwarg is preserved unchanged."""
        from giskard.llm.providers.azure_ai import AzureAIProvider

        monkeypatch.delenv("AZURE_AI_ENDPOINT", raising=False)

        with patch("openai.AsyncAzureOpenAI") as mock_client:
            AzureAIProvider(
                api_key="k",
                base_url="https://dev.services.ai.azure.com/openai/v1",
            )

        assert (
            mock_client.call_args.kwargs["azure_endpoint"]
            == "https://dev.services.ai.azure.com/openai/v1"
        )
