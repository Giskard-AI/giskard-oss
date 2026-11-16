"""Google Gemini ``generateContent`` translation tests.

Content shape: https://ai.google.dev/api/generate-content#Content
"""

import logging
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.routing import LLMClient
from giskard.llm.translators.google_chat import (
    _SKIP_THOUGHT_SIGNATURE,
    GoogleChatTranslator,
)
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    DeveloperMessage,
    FunctionMessage,
    RefusalContent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolCallFunction,
    UserMessage,
)

from .sdk_payload_validation import (
    validate_google_contents,
    validate_google_generate_content_config,
)
from .tool_turn_fixtures import (
    ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
    GET_TIME_TOOL,
    PARALLEL_TOOLS,
    PARALLEL_USER_PROMPT,
    TOOL_RESULT_CONTENT,
    TOOL_RESULT_TIME_PARALLEL,
    TOOL_RESULT_WEATHER_PARALLEL,
    WEATHER_TOOL,
    user_assistant_tool_then_tool_result,
    user_message_two_parallel_tool_calls_two_results,
    user_two_parallel_tool_calls_two_results,
)

_MODEL = "gemini-3.5-flash"


def test_single_user_message():
    """A lone user turn maps to one user ``contents`` entry."""
    msg: UserMessage = UserMessage(content="Hello.")
    payload = GoogleChatTranslator.to_google(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert payload.get("config", {}) == {}
    validate_google_contents(payload["contents"])


def test_single_user_message_with_text_content():
    """A lone user turn maps to one user ``contents`` entry."""
    msg: UserMessage = UserMessage(content=[TextContent(text="Hello.")])
    payload = GoogleChatTranslator.to_google(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert payload.get("config", {}) == {}
    validate_google_contents(payload["contents"])


def test_single_user_message_with_text_contents():
    """A lone user turn maps to one user ``contents`` entry."""
    msg: UserMessage = UserMessage(
        content=[TextContent(text="Hello."), TextContent(text="World.")]
    )
    payload = GoogleChatTranslator.to_google(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Hello."}, {"text": "World."}]}
    ]
    assert payload.get("config", {}) == {}
    validate_google_contents(payload["contents"])


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_instruction_then_user(instruction_role: Literal["system", "developer"]):
    """System or developer text becomes ``system_instruction``; user stays in ``contents``."""
    first: SystemMessage | DeveloperMessage = (
        SystemMessage(content="You are helpful.")
        if instruction_role == "system"
        else DeveloperMessage(content="You are helpful.")
    )
    messages: list[ChatMessage] = [
        first,
        UserMessage(content="Hello."),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("system_instruction") == ["You are helpful."]
    validate_google_contents(payload["contents"])


def test_system_then_developer_then_user():
    """System and developer preserve message order in ``system_instruction``."""
    messages: list[ChatMessage] = [
        SystemMessage(content="You are helpful."),
        DeveloperMessage(content="App version 2.0"),
        UserMessage(content="Hello."),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("system_instruction") == [
        "You are helpful.",
        "App version 2.0",
    ]
    validate_google_contents(payload["contents"])


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_two_instructions_then_user(instruction_role: Literal["system", "developer"]):
    """Two system or developer messages concatenate in order in ``system_instruction``."""
    messages: list[ChatMessage]
    if instruction_role == "system":
        messages = [
            SystemMessage(content="First system instruction."),
            SystemMessage(content="Second system instruction."),
            UserMessage(content="Hello."),
        ]
    else:
        messages = [
            DeveloperMessage(content="First system instruction."),
            DeveloperMessage(content="Second system instruction."),
            UserMessage(content="Hello."),
        ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("system_instruction") == [
        "First system instruction.",
        "Second system instruction.",
    ]
    validate_google_contents(payload["contents"])


def test_user_assistant_user():
    """User and model turns map to ``user`` / ``model`` content entries; ``reasoning`` is not replayed."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply.", reasoning="Private reasoning."),
        UserMessage(content="Second user."),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "First user."}]},
        {"role": "model", "parts": [{"text": "Assistant reply."}]},
        {"role": "user", "parts": [{"text": "Second user."}]},
    ]
    validate_google_contents(payload["contents"])


def test_assistant_refusal_replayed_as_text_part():
    """Gemini has no refusal part on input; ``refusal`` is sent as a ``text`` part."""
    messages: list[ChatMessage] = [
        UserMessage(content="Unsafe ask."),
        AssistantMessage(refusal="I can't help with that."),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Unsafe ask."}]},
        {"role": "model", "parts": [{"text": "I can't help with that."}]},
    ]
    validate_google_contents(payload["contents"])


def test_assistant_structured_refusal_parts_as_text():
    """Structured refusal parts map to plain ``text`` parts for Gemini."""
    messages: list[ChatMessage] = [
        AssistantMessage(
            content=[
                TextContent(text="Ok."),
                RefusalContent(refusal="Stopped."),
            ],
        ),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)
    assert payload["contents"] == [
        {
            "role": "model",
            "parts": [{"text": "Ok."}, {"text": "Stopped."}],
        },
    ]
    validate_google_contents(payload["contents"])


def test_user_tool_call_and_result_with_tools():
    """Tool declarations plus [user, model function_call, user function_response]."""
    messages = user_assistant_tool_then_tool_result()
    payload = GoogleChatTranslator.to_google(_MODEL, messages, tools=[WEATHER_TOOL])

    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("tools") == [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": WEATHER_TOOL.function.parameters,
                }
            ],
        },
    ]
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "What's the weather in Paris?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                    "thought_signature": _SKIP_THOUGHT_SIGNATURE,
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"result": TOOL_RESULT_CONTENT},
                    }
                }
            ],
        },
    ]
    validate_google_contents(payload["contents"])


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Two ``function_call`` parts on one model turn; two user ``function_response`` turns."""
    messages = user_two_parallel_tool_calls_two_results()
    payload = GoogleChatTranslator.to_google(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("tools") == [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": WEATHER_TOOL.function.parameters,
                }
            ],
        },
        {
            "function_declarations": [
                {
                    "name": "get_local_time",
                    "description": "Get local time for an IANA timezone.",
                    "parameters": GET_TIME_TOOL.function.parameters,
                }
            ],
        },
    ]
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": PARALLEL_USER_PROMPT}]},
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                    "thought_signature": _SKIP_THOUGHT_SIGNATURE,
                },
                {
                    "function_call": {
                        "name": "get_local_time",
                        "args": {"timezone": "Asia/Tokyo"},
                    },
                    "thought_signature": _SKIP_THOUGHT_SIGNATURE,
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"result": TOOL_RESULT_WEATHER_PARALLEL},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_local_time",
                        "response": {"result": TOOL_RESULT_TIME_PARALLEL},
                    }
                }
            ],
        },
    ]
    validate_google_contents(payload["contents"])


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Model turn mixes visible text with two parallel ``function_call`` parts."""
    messages = user_message_two_parallel_tool_calls_two_results()
    payload = GoogleChatTranslator.to_google(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("tools") == [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": WEATHER_TOOL.function.parameters,
                }
            ],
        },
        {
            "function_declarations": [
                {
                    "name": "get_local_time",
                    "description": "Get local time for an IANA timezone.",
                    "parameters": GET_TIME_TOOL.function.parameters,
                }
            ],
        },
    ]
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": PARALLEL_USER_PROMPT}]},
        {
            "role": "model",
            "parts": [
                {"text": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS},
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                    "thought_signature": _SKIP_THOUGHT_SIGNATURE,
                },
                {
                    "function_call": {
                        "name": "get_local_time",
                        "args": {"timezone": "Asia/Tokyo"},
                    },
                    "thought_signature": _SKIP_THOUGHT_SIGNATURE,
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"result": TOOL_RESULT_WEATHER_PARALLEL},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_local_time",
                        "response": {"result": TOOL_RESULT_TIME_PARALLEL},
                    }
                }
            ],
        },
    ]
    validate_google_contents(payload["contents"])


def test_assistant_text_thought_signature_is_replayed():
    """A captured text-part ``thought_signature`` is replayed verbatim."""
    messages: list[ChatMessage] = [
        UserMessage(content="Hi."),
        AssistantMessage(
            content=[
                TextContent(
                    text="Thinking about it.",
                    thought_signature=b"text-signature-bytes",
                )
            ],
        ),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Hi."}]},
        {
            "role": "model",
            "parts": [
                {
                    "text": "Thinking about it.",
                    "thought_signature": b"text-signature-bytes",
                }
            ],
        },
    ]
    validate_google_contents(payload["contents"])


def test_tool_call_thought_signature_is_replayed():
    """A captured ``thought_signature`` is replayed verbatim, not the skip sentinel."""
    messages: list[ChatMessage] = [
        UserMessage(content="What's the weather in Paris?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="call_weather_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments={"city": "Paris"},
                    ),
                    thought_signature=b"real-signature-bytes",
                )
            ],
        ),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages, tools=[WEATHER_TOOL])
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "What's the weather in Paris?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                    "thought_signature": b"real-signature-bytes",
                }
            ],
        },
    ]
    validate_google_contents(payload["contents"])


def test_function_message_raises():
    """FunctionMessage is not supported by the Google translator."""
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        FunctionMessage(name="fn", content="result"),
    ]
    with pytest.raises(ValueError, match="Unsupported message role"):
        GoogleChatTranslator.to_google(_MODEL, messages)


@pytest.mark.parametrize(
    ("raw_effort", "expected_level"),
    [
        pytest.param("minimal", "MINIMAL", id="minimal"),
        pytest.param("low", "LOW", id="low"),
        pytest.param("medium", "MEDIUM", id="medium"),
        pytest.param("high", "HIGH", id="high"),
        pytest.param("HIGH", "HIGH", id="upper-case"),
        pytest.param("MeDiUm", "MEDIUM", id="mixed-case"),
    ],
)
def test_reasoning_effort_supported_levels_emit_upper_thinking_level(
    raw_effort: str, expected_level: str, caplog: pytest.LogCaptureFixture
):
    """Supported levels set ``config.thinking_config.thinking_level`` in uppercase."""
    with caplog.at_level(logging.WARNING, logger="giskard.llm.translators.google_chat"):
        payload = GoogleChatTranslator.to_google(
            _MODEL, [UserMessage(content="Hi.")], reasoning_effort=raw_effort
        )
    cfg = payload.get("config", {})
    assert "reasoning_effort" not in cfg
    assert cfg.get("thinking_config") == {"thinking_level": expected_level}
    assert "reasoning_effort" not in " ".join(r.message for r in caplog.records)
    validate_google_contents(payload["contents"])
    validate_google_generate_content_config(cfg)


def test_reasoning_effort_none_omitted():
    """Explicit ``reasoning_effort=None`` omits ``thinking_config``."""
    payload = GoogleChatTranslator.to_google(
        _MODEL, [UserMessage(content="Hi.")], reasoning_effort=None
    )
    assert payload.get("config", {}) == {}


def test_thinking_config_escape_hatch_merges_with_reasoning_effort(
    caplog: pytest.LogCaptureFixture,
):
    """``reasoning_effort`` merges into ``thinking_config`` preserving sibling keys."""
    escape = {"include_thoughts": True}
    with caplog.at_level(logging.WARNING, logger="giskard.llm.translators.google_chat"):
        payload = GoogleChatTranslator.to_google(
            _MODEL,
            [UserMessage(content="Hi.")],
            reasoning_effort="low",
            thinking_config=escape,
        )
    cfg = payload.get("config", {})
    assert cfg.get("thinking_config") == {
        "include_thoughts": True,
        "thinking_level": "LOW",
    }
    assert escape == {"include_thoughts": True}
    assert "thinking_config" not in " ".join(r.message for r in caplog.records)
    validate_google_generate_content_config(cfg)


def test_thinking_config_escape_hatch_only_passes_through():
    """Explicit ``thinking_config`` (e.g. legacy ``thinking_budget=0``) passes through."""
    payload = GoogleChatTranslator.to_google(
        _MODEL,
        [UserMessage(content="Hi.")],
        thinking_config={"thinking_budget": 0, "include_thoughts": False},
    )
    cfg = payload.get("config", {})
    assert cfg.get("thinking_config") == {
        "thinking_budget": 0,
        "include_thoughts": False,
    }
    validate_google_generate_content_config(cfg)


@pytest.mark.parametrize(
    "invalid_effort",
    [
        pytest.param("none", id="unsupported-none"),
        pytest.param("xhigh", id="unsupported-xhigh"),
        pytest.param("max", id="unsupported-max"),
        pytest.param("MAX", id="unsupported-max-upper"),
        pytest.param("ultra", id="invalid-string"),
        pytest.param("", id="empty-string"),
    ],
)
def test_reasoning_effort_rejects_unsupported_and_invalid_at_translator_seam(
    invalid_effort: str,
):
    """Unsupported and invalid levels raise ``BadRequestError(400, ..., 'google')``."""
    with pytest.raises(BadRequestError) as exc_info:
        GoogleChatTranslator.to_google(
            _MODEL, [UserMessage(content="Hi.")], reasoning_effort=invalid_effort
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.provider == "google"


@pytest.mark.parametrize(
    "conflicting_thinking_config",
    [
        pytest.param({"thinking_level": "LOW"}, id="snake-thinking-level"),
        pytest.param({"thinkingLevel": "LOW"}, id="camel-thinking-level"),
        pytest.param({"thinking_budget": 1024}, id="snake-thinking-budget-positive"),
        pytest.param({"thinking_budget": 0}, id="snake-thinking-budget-zero"),
        pytest.param({"thinkingBudget": 512}, id="camel-thinking-budget"),
    ],
)
def test_reasoning_effort_and_thinking_config_conflict_raises(
    conflicting_thinking_config: dict[str, Any],
):
    """Passing both ``reasoning_effort`` and native level/budget keys raises 400."""
    with pytest.raises(BadRequestError) as exc_info:
        GoogleChatTranslator.to_google(
            _MODEL,
            [UserMessage(content="Hi.")],
            reasoning_effort="high",
            thinking_config=conflicting_thinking_config,
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.provider == "google"


@pytest.mark.parametrize("prefix", ["google", "gemini"])
async def test_client_acompletion_google_forwards_and_rejects_before_sdk(
    prefix: str, monkeypatch: pytest.MonkeyPatch
):
    """``LLMClient.acompletion`` forwards merged thinking_config and rejects 400 before SDK."""
    from google.genai import types

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = LLMClient()
    provider = client._get_provider(prefix)
    sdk_resp = types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {"content": {"parts": [{"text": "Ok"}]}, "finish_reason": "STOP"}
            ],
            "usage_metadata": {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "total_token_count": 2,
            },
        }
    )
    generate_mock = AsyncMock(return_value=sdk_resp)
    cast(Any, provider)._client = MagicMock(
        aio=MagicMock(models=MagicMock(generate_content=generate_mock))
    )

    await client.acompletion(
        f"{prefix}/{_MODEL}",
        [{"role": "user", "content": "Hi"}],
        reasoning_effort="HIGH",
        thinking_config={"include_thoughts": True},
    )
    assert generate_mock.call_args.kwargs["config"]["thinking_config"] == {
        "include_thoughts": True,
        "thinking_level": "HIGH",
    }

    bad_cases: tuple[dict[str, Any], ...] = (
        {"reasoning_effort": "none"},
        {"reasoning_effort": "xhigh"},
        {"reasoning_effort": "max"},
        {"reasoning_effort": "ultra"},
        {"reasoning_effort": "high", "thinking_config": {"thinking_budget": 0}},
        {"reasoning_effort": "high", "thinking_config": {"thinkingLevel": "LOW"}},
    )
    for bad_kwargs in bad_cases:
        generate_mock.reset_mock()
        with pytest.raises(BadRequestError) as exc_info:
            await client.acompletion(
                f"{prefix}/{_MODEL}",
                [{"role": "user", "content": "Hi"}],
                **bad_kwargs,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider == "google"
        generate_mock.assert_not_called()
