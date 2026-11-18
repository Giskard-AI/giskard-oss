"""Unit tests for the optional ``LiteLLMGenerator``.

These tests exercise the LiteLLM adapter in isolation (no network) with a
mocked ``litellm.acompletion``. They are marked ``litellm`` so they are
automatically skipped when the optional ``litellm`` extra is not installed.
"""

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.agents.generators.litellm_generator import (
    LiteLLMGenerator,
    LiteLLMRetryMiddleware,
    RetryPolicy,
)
from giskard.llm.types import (
    AssistantMessage,
    ReasoningContent,
    ReasoningText,
    TextContent,
    UserMessage,
    UserMessageParam,
)
from pydantic import BaseModel

litellm = pytest.importorskip("litellm")

pytestmark = pytest.mark.litellm

_USER_MESSAGE: list[UserMessageParam] = [{"role": "user", "content": "Hi"}]


def _make_litellm_response(content: str = "Mock response") -> Any:
    """Build a minimal ``litellm.ModelResponse``-shaped mock."""
    msg = MagicMock()
    msg.model_dump.return_value = {"role": "assistant", "content": content}
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    raw = MagicMock()
    raw.choices = [choice]
    raw.model_dump.return_value = {"id": "mock-id", "model": "mock-model"}
    return raw


@pytest.fixture
def mock_acompletion() -> Any:
    """Patch ``litellm.acompletion`` with a canned response and yield the mock."""
    with patch("litellm.acompletion", return_value=_make_litellm_response()) as mock:
        yield mock


async def test_litellm_generator_completion(mock_acompletion: Any) -> None:
    """``LiteLLMGenerator._call_model`` routes through ``litellm.acompletion``."""
    generator = LiteLLMGenerator(model="gemini/gemini-3.5-flash")
    response = await generator.complete(messages=_USER_MESSAGE)

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Mock response"
    assert response.choices[0].finish_reason == "stop"
    mock_acompletion.assert_called_once()
    assert mock_acompletion.call_args.kwargs["model"] == "gemini/gemini-3.5-flash"


async def test_litellm_generator_forwards_params(mock_acompletion: Any) -> None:
    """Generator params and call-site params are merged and forwarded."""
    generator = LiteLLMGenerator(model="test-model").with_params(temperature=0.3)
    await generator.complete(
        messages=_USER_MESSAGE, params=GenerationParams(max_tokens=50)
    )

    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 50


@pytest.mark.parametrize(
    ("source", "explicit_temperature"),
    [
        pytest.param("constructor", None, id="constructor-omit-temp"),
        pytest.param("with_params", None, id="with-params-omit-temp"),
        pytest.param("call_params", None, id="call-params-omit-temp"),
        pytest.param("with_params", 0.4, id="with-params-explicit-temp"),
    ],
)
async def test_litellm_generator_forwards_reasoning_effort_and_temperature(
    mock_acompletion: Any,
    source: str,
    explicit_temperature: float | None,
) -> None:
    """``LiteLLMGenerator`` forwards normalized ``reasoning_effort`` and omits unset ``temperature``."""
    extra: dict[str, Any] = (
        {"temperature": explicit_temperature}
        if explicit_temperature is not None
        else {}
    )
    if source == "constructor":
        generator = LiteLLMGenerator(
            model="test-model",
            params=GenerationParams.model_validate(
                {"reasoning_effort": "HIGH", **extra}
            ),
        )
        call_params = None
    elif source == "with_params":
        generator = LiteLLMGenerator(model="test-model").with_params(
            reasoning_effort="HIGH", **extra
        )
        call_params = None
    else:
        generator = LiteLLMGenerator(model="test-model")
        call_params = GenerationParams.model_validate(
            {"reasoning_effort": "HIGH", **extra}
        )

    await generator.complete(messages=_USER_MESSAGE, params=call_params)

    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["reasoning_effort"] == "high"
    if explicit_temperature is None:
        assert "temperature" not in kwargs
    else:
        assert kwargs["temperature"] == explicit_temperature


@pytest.mark.parametrize(
    "clear_via",
    [
        pytest.param("with_params", id="clear-via-with-params"),
        pytest.param("call_params", id="clear-via-call-params"),
    ],
)
async def test_litellm_generator_clear_reasoning_effort_with_none(
    mock_acompletion: Any,
    clear_via: str,
) -> None:
    """Explicit ``reasoning_effort=None`` clears an earlier setting without error."""
    generator = LiteLLMGenerator(model="test-model").with_params(
        reasoning_effort="high"
    )
    if clear_via == "with_params":
        generator = generator.with_params(reasoning_effort=None)
        call_params = None
    else:
        call_params = GenerationParams(reasoning_effort=None)

    await generator.complete(messages=_USER_MESSAGE, params=call_params)

    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs.get("reasoning_effort") is None
    assert "temperature" not in kwargs


async def test_litellm_generator_sanitizes_response_format_schema_name(
    mock_acompletion: Any,
) -> None:
    """Bracketed generic model names are sanitized to a provider-safe schema name."""

    class _Wrapper[T](BaseModel):
        value: T | None = None

    assert _Wrapper[str].__name__ == "_Wrapper[str]"

    generator = LiteLLMGenerator(model="azure/gpt-4o")
    await generator.complete(
        messages=_USER_MESSAGE,
        params=GenerationParams(response_format=_Wrapper[str]),
    )

    response_format = mock_acompletion.call_args.kwargs["response_format"]
    assert isinstance(response_format, dict)
    name = response_format["json_schema"]["name"]
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", name), name


async def test_litellm_generator_does_not_send_reasoning(
    mock_acompletion: Any,
) -> None:
    """Reasoning parts and the deprecated ``reasoning`` field are never sent.

    Chat Completions has no reasoning field (ADR-0003), whatever the route.
    """
    generator = LiteLLMGenerator(model="azure/gpt-4o")
    await generator.complete(
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(
                content=[
                    ReasoningContent(
                        content=[ReasoningText(text="Greet back.")],
                        signature="s",
                        provider="anthropic",
                    ),
                    TextContent(text="Hello."),
                ],
                reasoning="Greet back.",
            ),
            UserMessage(content="Bye"),
        ]
    )

    sent = mock_acompletion.call_args.kwargs["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
    }


@pytest.mark.parametrize(
    "status_code, expected",
    [
        pytest.param(429, True, id="rate-limit-retryable"),
        pytest.param(500, True, id="server-error-retryable"),
        pytest.param(400, False, id="bad-request-not-retryable"),
    ],
)
def test_litellm_retry_middleware_should_retry(
    status_code: int, expected: bool
) -> None:
    """Retry middleware defers to ``litellm._should_retry`` via ``status_code``."""
    mw = LiteLLMRetryMiddleware(retry_policy=RetryPolicy(max_attempts=2))
    err = Exception("boom")
    err.status_code = status_code  # pyright: ignore[reportAttributeAccessIssue]

    assert mw._should_retry(err) is expected


def test_litellm_generator_registered_as_kind() -> None:
    """The real ``LiteLLMGenerator`` owns the ``"litellm"`` discriminator kind."""
    instance = LiteLLMGenerator(model="gemini/gemini-3.5-flash")
    dumped = instance.model_dump()
    assert dumped["kind"] == "litellm"

    reconstructed = BaseGenerator.model_validate(dumped)
    assert isinstance(reconstructed, LiteLLMGenerator)
    assert reconstructed.model == "gemini/gemini-3.5-flash"
