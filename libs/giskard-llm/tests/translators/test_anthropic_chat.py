"""Anthropic Messages API translation tests.

Request shape: https://docs.anthropic.com/en/api/messages
"""

import logging
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.routing import LLMClient
from giskard.llm.translators.anthropic import AnthropicChatTranslator
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    DeveloperMessage,
    FunctionMessage,
    ReasoningContent,
    ReasoningEffort,
    ReasoningSummary,
    ReasoningText,
    RefusalContent,
    SystemMessage,
    TextContent,
    UserMessage,
)

from .nested_schema_models import NestedOutputModel
from .sdk_payload_validation import validate_anthropic_message_create
from .tool_turn_fixtures import (
    ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
    GET_TIME_TOOL,
    PARALLEL_TOOLS,
    PARALLEL_USER_PROMPT,
    TOOL_CALL_ID,
    TOOL_CALL_ID_TIME_PARALLEL,
    TOOL_CALL_ID_WEATHER_PARALLEL,
    TOOL_RESULT_CONTENT,
    TOOL_RESULT_TIME_PARALLEL,
    TOOL_RESULT_WEATHER_PARALLEL,
    WEATHER_TOOL,
    user_assistant_tool_then_tool_result,
    user_message_two_parallel_tool_calls_two_results,
    user_two_parallel_tool_calls_two_results,
)

_MODEL = "claude-sonnet-4-20250514"


def _thinking(text: str, signature: str, provider: str | None = "anthropic"):
    return ReasoningContent(
        content=[ReasoningText(text=text)], signature=signature, provider=provider
    )


def _redacted(data: str, provider: str | None = "anthropic"):
    return ReasoningContent(encrypted_content=data, provider=provider)


def test_single_user_message():
    """A lone user turn maps to one ``messages`` entry (typical conversation start)."""
    msg: UserMessage = UserMessage(content="Hello.")
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["max_tokens"] == 4096
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    assert "system" not in payload
    validate_anthropic_message_create(payload)


def test_single_user_message_with_text_content():
    """A lone user turn maps to one ``messages`` entry (typical conversation start)."""
    msg: UserMessage = UserMessage(content=[TextContent(text="Hello.")])
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["max_tokens"] == 4096
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hello."}]}
    ]
    assert "system" not in payload
    validate_anthropic_message_create(payload)


def test_single_user_message_with_text_contents():
    """A lone user turn maps to one ``messages`` entry (typical conversation start)."""
    msg: UserMessage = UserMessage(
        content=[TextContent(text="Hello."), TextContent(text="World.")]
    )
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["max_tokens"] == 4096
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello."},
                {"type": "text", "text": "World."},
            ],
        }
    ]
    assert "system" not in payload
    validate_anthropic_message_create(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_instruction_then_user(instruction_role: Literal["system", "developer"]):
    """System or developer text maps to top-level ``system`` blocks; user to ``messages``."""
    first: SystemMessage | DeveloperMessage = (
        SystemMessage(content="You are helpful.")
        if instruction_role == "system"
        else DeveloperMessage(content="You are helpful.")
    )
    messages: list[ChatMessage] = [
        first,
        UserMessage(content="Hello."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload.get("system") == [{"type": "text", "text": "You are helpful."}]
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    validate_anthropic_message_create(payload)


def test_system_then_developer_then_user():
    """System and developer preserve order in top-level ``system`` blocks."""
    messages: list[ChatMessage] = [
        SystemMessage(content="You are helpful."),
        DeveloperMessage(content="App version 2.0"),
        UserMessage(content="Hello."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload.get("system") == [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": "App version 2.0"},
    ]
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    validate_anthropic_message_create(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_two_instructions_then_user(instruction_role: Literal["system", "developer"]):
    """Two system or developer messages become several ``system`` text blocks in order."""
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
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload.get("system") == [
        {"type": "text", "text": "First system instruction."},
        {"type": "text", "text": "Second system instruction."},
    ]
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    validate_anthropic_message_create(payload)


def test_user_assistant_user():
    """Multi-turn chat: a signed Anthropic reasoning part is replayed as ``thinking``."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(
            content=[
                _thinking("Private reasoning.", "sig_1"),
                TextContent(text="Assistant reply."),
            ],
        ),
        UserMessage(content="Second user."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "user", "content": "First user."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Private reasoning.",
                    "signature": "sig_1",
                },
                {"type": "text", "text": "Assistant reply."},
            ],
        },
        {"role": "user", "content": "Second user."},
    ]
    validate_anthropic_message_create(payload)


def test_plain_reasoning_string_is_not_replayed():
    """The deprecated ``reasoning`` string is never a replay source."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply.", reasoning="Private reasoning."),
        UserMessage(content="Second user."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Assistant reply."}],
    }
    validate_anthropic_message_create(payload)


def test_interleaved_reasoning_replayed_in_stored_order_then_tool_use():
    """Content parts are emitted in stored order, then tool calls."""
    messages = user_assistant_tool_then_tool_result()
    assistant = cast(AssistantMessage, messages[1])
    messages[1] = assistant.model_copy(
        update={
            "content": [
                _thinking("First.", "sig_1"),
                TextContent(text="Checking."),
                _redacted("opaque"),
                _thinking("Second.", "sig_2"),
                TextContent(text="Still checking."),
            ],
        }
    )
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        messages,
        tools=[WEATHER_TOOL],
        thinking={"type": "adaptive"},
    )

    assert payload["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "First.", "signature": "sig_1"},
            {"type": "text", "text": "Checking."},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "thinking", "thinking": "Second.", "signature": "sig_2"},
            {"type": "text", "text": "Still checking."},
            {
                "type": "tool_use",
                "id": TOOL_CALL_ID,
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
    }
    validate_anthropic_message_create(payload)


def test_empty_signed_thinking_replays_as_thinking_not_redacted():
    """Empty text with a signature is ``display: "omitted"``: still a ``thinking`` block."""
    messages: list[ChatMessage] = [
        UserMessage(content="Hi."),
        AssistantMessage(
            content=[_thinking("", "sig_omitted"), TextContent(text="Hello.")]
        ),
        UserMessage(content="Again."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig_omitted"},
        {"type": "text", "text": "Hello."},
    ]
    validate_anthropic_message_create(payload)


def test_multi_text_reasoning_part_joins_texts():
    """A part with several ``reasoning_text`` entries replays their joined text."""
    part = ReasoningContent(
        content=[ReasoningText(text="One."), ReasoningText(text="Two.")],
        signature="sig",
        provider="anthropic",
    )
    messages: list[ChatMessage] = [
        UserMessage(content="Hi."),
        AssistantMessage(content=[part, TextContent(text="Hello.")]),
        UserMessage(content="Again."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert list(payload["messages"][1]["content"])[0] == {
        "type": "thinking",
        "thinking": "One.\nTwo.",
        "signature": "sig",
    }


def test_unreplayable_reasoning_parts_dropped():
    """Only signed or redacted parts tagged ``anthropic`` are replayed."""
    messages: list[ChatMessage] = [
        UserMessage(content="Hi."),
        AssistantMessage(
            content=[
                ReasoningContent(
                    content=[ReasoningText(text="Unsigned.")], provider="anthropic"
                ),
                _thinking("Foreign.", "sig", provider="google"),
                _redacted("enc", provider="openai"),
                _thinking("Untagged.", "sig", provider=None),
                ReasoningContent(
                    summary=[ReasoningSummary(text="Summary.")], provider="anthropic"
                ),
                _thinking("Kept.", "sig_k"),
                TextContent(text="Hello."),
            ],
        ),
        UserMessage(content="Again."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "Kept.", "signature": "sig_k"},
        {"type": "text", "text": "Hello."},
    ]
    validate_anthropic_message_create(payload)


def test_merged_assistant_turns_keep_thinking_first():
    """Merging consecutive assistant messages hoists all thinking to the front,
    keeping the relative order of thinking blocks and of the other blocks."""
    messages: list[ChatMessage] = [
        UserMessage(content="Hi."),
        AssistantMessage(
            content=[_thinking("A.", "sig_a"), TextContent(text="Part one.")]
        ),
        AssistantMessage(
            content=[
                _redacted("opaque_b"),
                _thinking("C.", "sig_c"),
                TextContent(text="Part two."),
            ]
        ),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "user", "content": "Hi."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "A.", "signature": "sig_a"},
                {"type": "redacted_thinking", "data": "opaque_b"},
                {"type": "thinking", "thinking": "C.", "signature": "sig_c"},
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
        },
    ]
    validate_anthropic_message_create(payload)


def test_thinking_param_passed_through_without_warning(caplog):
    """``thinking`` is a known Anthropic param: forwarded as-is, no unknown-param warning."""
    with caplog.at_level(logging.WARNING):
        payload = AnthropicChatTranslator.to_anthropic(
            _MODEL, [UserMessage(content="Hi.")], thinking={"type": "adaptive"}
        )
    assert payload.get("thinking") == {"type": "adaptive"}
    assert "thinking" not in " ".join(r.message for r in caplog.records)
    validate_anthropic_message_create(payload)


def test_assistant_refusal_replayed_as_text_block():
    """Anthropic has no refusal part on input; we replay ``refusal`` as a normal text block."""
    messages: list[ChatMessage] = [
        UserMessage(content="Unsafe ask."),
        AssistantMessage(refusal="I can't help with that."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)
    assert payload["messages"] == [
        {"role": "user", "content": "Unsafe ask."},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I can't help with that."}],
        },
    ]
    validate_anthropic_message_create(payload)


def test_assistant_structured_refusal_parts_as_text():
    """``TextContentParam`` / ``RefusalContentParam`` lists become text blocks (no distinct refusal type)."""
    messages: list[ChatMessage] = [
        AssistantMessage(
            content=[TextContent(text="Ok."), RefusalContent(refusal="Stopped.")],
        ),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)
    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Ok."},
                {"type": "text", "text": "Stopped."},
            ],
        },
    ]
    validate_anthropic_message_create(payload)


def test_user_tool_call_and_result_with_tools():
    """Tools plus [user, assistant tool_use, user tool_result] message shape."""
    messages = user_assistant_tool_then_tool_result()
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, messages, tools=[WEATHER_TOOL]
    )

    assert payload.get("tools") == [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": WEATHER_TOOL.function.parameters,
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID,
                    "content": [
                        {
                            "type": "text",
                            "text": TOOL_RESULT_CONTENT,
                        }
                    ],
                }
            ],
        },
    ]
    validate_anthropic_message_create(payload)


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Two ``tool_use`` blocks on one assistant turn; adjacent ``tool`` turns merge to one user."""
    messages = user_two_parallel_tool_calls_two_results()
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, messages, tools=PARALLEL_TOOLS
    )

    assert payload.get("tools") == [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": WEATHER_TOOL.function.parameters,
        },
        {
            "name": "get_local_time",
            "description": "Get local time for an IANA timezone.",
            "input_schema": GET_TIME_TOOL.function.parameters,
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "name": "get_local_time",
                    "input": {"timezone": "Asia/Tokyo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "content": [
                        {
                            "type": "text",
                            "text": TOOL_RESULT_WEATHER_PARALLEL,
                        }
                    ],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID_TIME_PARALLEL,
                    "content": [
                        {
                            "type": "text",
                            "text": TOOL_RESULT_TIME_PARALLEL,
                        }
                    ],
                },
            ],
        },
    ]
    validate_anthropic_message_create(payload)


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Assistant turn with leading text plus two parallel ``tool_use`` blocks."""
    messages = user_message_two_parallel_tool_calls_two_results()
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, messages, tools=PARALLEL_TOOLS
    )

    assert payload.get("tools") == [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": WEATHER_TOOL.function.parameters,
        },
        {
            "name": "get_local_time",
            "description": "Get local time for an IANA timezone.",
            "input_schema": GET_TIME_TOOL.function.parameters,
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS},
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "name": "get_local_time",
                    "input": {"timezone": "Asia/Tokyo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "content": [
                        {
                            "type": "text",
                            "text": TOOL_RESULT_WEATHER_PARALLEL,
                        }
                    ],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": TOOL_CALL_ID_TIME_PARALLEL,
                    "content": [
                        {
                            "type": "text",
                            "text": TOOL_RESULT_TIME_PARALLEL,
                        }
                    ],
                },
            ],
        },
    ]
    validate_anthropic_message_create(payload)


def test_function_message_raises():
    """FunctionMessage is not supported by the Anthropic translator."""
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        FunctionMessage(name="fn", content="result"),
    ]
    with pytest.raises(ValueError, match="Unsupported message role"):
        AnthropicChatTranslator.to_anthropic(_MODEL, messages)


def test_unknown_completion_params_warn(caplog):
    """Unsupported kwargs are dropped but callers get a warning (#2613)."""
    msg = UserMessage(content="Hello.")
    with caplog.at_level(logging.WARNING):
        payload = AnthropicChatTranslator.to_anthropic(
            _MODEL, [msg], top_p=0.5, stop_sequences=["STOP"], temperature=0.2
        )
    assert "temperature" not in payload
    assert payload.get("extra_body") == {"temperature": 0.2}
    assert "top_p" not in payload
    assert payload.get("stop_sequences") == ["STOP"]
    joined = " ".join(r.message for r in caplog.records)
    assert "top_p" in joined
    assert "stop_sequences" not in joined


def test_sdk_v1_messages_create_accepts_translator_payload():
    """Unmocked SDK check: v1 rejects top-level temperature but accepts extra_body."""
    pytest.importorskip("anthropic")
    import inspect

    from anthropic.resources.messages.messages import AsyncMessages

    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hello.")],
        temperature=0.2,
        stop_sequences=["STOP"],
    )
    assert "temperature" not in payload
    assert payload.get("extra_body") == {"temperature": 0.2}
    assert payload.get("stop_sequences") == ["STOP"]
    inspect.signature(AsyncMessages.create).bind_partial(**payload)


def test_httpx_v1_timeout_converted_to_httpx2():
    """An httpx (v1) Timeout is converted, not mis-parsed as a scalar by httpx2."""
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("httpx2")
    from httpx2 import Timeout

    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hello.")],
        timeout=httpx.Timeout(5.0, connect=2.0),
    )
    timeout = payload.get("timeout")
    assert isinstance(timeout, Timeout)
    assert timeout.connect == 2.0
    assert timeout.read == 5.0
    assert timeout.write == 5.0
    assert timeout.pool == 5.0


def test_httpx2_timeout_passes_through():
    """A native httpx2 Timeout is forwarded untouched."""
    httpx2 = pytest.importorskip("httpx2")

    original = httpx2.Timeout(5.0, connect=2.0)
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, [UserMessage(content="Hello.")], timeout=original
    )
    assert payload.get("timeout") is original


def test_response_format_nested_pydantic_schema_valid_for_anthropic():
    """Nested ``$defs`` get ``additionalProperties: false`` on every object (#2859)."""
    pytest.importorskip("anthropic")
    msg = UserMessage(content="hi")
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, [msg], response_format=NestedOutputModel
    )
    output_config = cast(dict[str, Any], cast(object, payload.get("output_config")))
    schema = cast(dict[str, Any], output_config["format"]["schema"])
    assert "$defs" in schema or schema.get("properties")
    assert schema["additionalProperties"] is False
    validate_anthropic_message_create(payload)


def test_response_format_preserves_const_and_pattern_constraints():
    pytest.importorskip("anthropic")
    from pydantic import BaseModel, Field

    class Input(BaseModel):
        language: Literal["en"]
        ticket: str = Field(pattern=r"^TICKET-[0-9]+$", min_length=1)

    class Output(BaseModel):
        inputs: list[Input]
        done: Literal[False]

    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, [UserMessage(content="hi")], response_format=Output
    )
    output = cast(dict[str, Any], cast(object, payload))
    schema = output["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["done"]["const"] is False
    inner = schema["$defs"]["Input"]
    assert inner["additionalProperties"] is False
    assert inner["properties"]["language"]["const"] == "en"
    assert inner["properties"]["ticket"]["pattern"] == r"^TICKET-[0-9]+$"
    assert "minLength" not in inner["properties"]["ticket"]
    validate_anthropic_message_create(payload)


def test_response_format_dict_uses_anthropic_output_config_shape():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="hi")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Result", "schema": schema},
        },
    )
    assert cast(dict[str, Any], cast(object, payload))["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }
    assert "response_format" not in payload
    validate_anthropic_message_create(payload)


@pytest.mark.parametrize(
    ("raw_effort", "expected"),
    [
        pytest.param("low", "low", id="low"),
        pytest.param("medium", "medium", id="medium"),
        pytest.param("high", "high", id="high"),
        pytest.param("xhigh", "xhigh", id="xhigh"),
        pytest.param("max", "max", id="max"),
        pytest.param("HIGH", "high", id="upper-case"),
        pytest.param("xHiGh", "xhigh", id="mixed-case"),
    ],
)
def test_reasoning_effort_supported_levels_set_output_config_and_adaptive_thinking(
    raw_effort: str, expected: ReasoningEffort, caplog: pytest.LogCaptureFixture
):
    """Supported levels set ``output_config['effort']`` and auto-inject adaptive thinking."""
    with caplog.at_level(logging.WARNING, logger="giskard.llm.translators.anthropic"):
        payload = AnthropicChatTranslator.to_anthropic(
            _MODEL, [UserMessage(content="Hi.")], reasoning_effort=raw_effort
        )
    assert "reasoning_effort" not in payload
    assert payload.get("output_config") == {"effort": expected}
    assert payload.get("thinking") == {"type": "adaptive"}
    assert "reasoning_effort" not in " ".join(r.message for r in caplog.records)
    validate_anthropic_message_create(payload)


def test_reasoning_effort_none_omitted_and_no_thinking_injected():
    """Explicit ``reasoning_effort=None`` omits ``output_config`` and does not inject ``thinking``."""
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, [UserMessage(content="Hi.")], reasoning_effort=None
    )
    assert "reasoning_effort" not in payload
    assert "output_config" not in payload
    assert "thinking" not in payload
    validate_anthropic_message_create(payload)


def test_reasoning_effort_merges_with_response_format_pydantic():
    """``output_config['effort']`` and ``output_config['format']`` from ``response_format`` coexist."""
    pytest.importorskip("anthropic")
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hi.")],
        reasoning_effort="HIGH",
        response_format=NestedOutputModel,
    )
    output_config = cast(dict[str, Any], cast(object, payload.get("output_config")))
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"
    assert payload.get("thinking") == {"type": "adaptive"}
    validate_anthropic_message_create(payload)


def test_reasoning_effort_merges_with_explicit_output_config_format():
    """``reasoning_effort`` merges into explicit ``output_config`` preserving ``format``."""
    output_config_in = {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        }
    }
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hi.")],
        reasoning_effort="medium",
        output_config=output_config_in,
    )
    assert payload.get("output_config") == {
        "format": output_config_in["format"],
        "effort": "medium",
    }
    assert "effort" not in output_config_in
    assert payload.get("thinking") == {"type": "adaptive"}
    validate_anthropic_message_create(payload)


def test_response_format_preserves_explicit_output_config_effort():
    """``response_format`` merges into explicit ``output_config`` without dropping ``effort``."""
    pytest.importorskip("anthropic")
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hi.")],
        output_config={"effort": "low"},
        response_format=NestedOutputModel,
    )
    output_config = cast(dict[str, Any], cast(object, payload.get("output_config")))
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"
    assert "thinking" not in payload
    validate_anthropic_message_create(payload)


def test_reasoning_effort_preserves_explicit_thinking():
    """Explicit ``thinking`` escape hatch is preserved untouched alongside ``reasoning_effort``."""
    explicit_thinking = {"type": "enabled", "budget_tokens": 2048}
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL,
        [UserMessage(content="Hi.")],
        reasoning_effort="high",
        thinking=explicit_thinking,
    )
    assert payload.get("thinking") == {"type": "enabled", "budget_tokens": 2048}
    assert payload.get("output_config") == {"effort": "high"}
    validate_anthropic_message_create(payload)


@pytest.mark.parametrize(
    "invalid_effort",
    [
        pytest.param("none", id="unsupported-none"),
        pytest.param("NONE", id="unsupported-none-upper"),
        pytest.param("minimal", id="unsupported-minimal"),
        pytest.param("ultra", id="invalid-string"),
        pytest.param("", id="empty-string"),
    ],
)
def test_reasoning_effort_rejects_unsupported_and_invalid_at_translator_seam(
    invalid_effort: str,
):
    """Unsupported and invalid levels raise ``BadRequestError(400, ..., 'anthropic')``."""
    with pytest.raises(BadRequestError) as exc_info:
        AnthropicChatTranslator.to_anthropic(
            _MODEL, [UserMessage(content="Hi.")], reasoning_effort=invalid_effort
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.provider == "anthropic"


def test_reasoning_effort_and_output_config_effort_conflict_raises():
    """Providing both ``reasoning_effort`` and ``output_config['effort']`` raises 400."""
    with pytest.raises(BadRequestError) as exc_info:
        AnthropicChatTranslator.to_anthropic(
            _MODEL,
            [UserMessage(content="Hi.")],
            reasoning_effort="high",
            output_config={"effort": "low"},
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.provider == "anthropic"


async def test_client_acompletion_anthropic_forwards_and_rejects_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
):
    """``LLMClient.acompletion`` forwards merged params and raises 400 before calling SDK."""
    from anthropic.types import Message

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = LLMClient()
    provider = client._get_provider("anthropic")
    sdk_resp = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": _MODEL,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": "Ok"}],
        }
    )
    create_mock = AsyncMock(return_value=sdk_resp)
    cast(Any, provider)._client = MagicMock(messages=MagicMock(create=create_mock))

    await client.acompletion(
        f"anthropic/{_MODEL}",
        [{"role": "user", "content": "Hi"}],
        reasoning_effort="HIGH",
    )
    assert create_mock.call_args.kwargs["output_config"] == {"effort": "high"}
    assert create_mock.call_args.kwargs["thinking"] == {"type": "adaptive"}

    bad_cases: tuple[dict[str, Any], ...] = (
        {"reasoning_effort": "none"},
        {"reasoning_effort": "minimal"},
        {"reasoning_effort": "ultra"},
        {"reasoning_effort": "high", "output_config": {"effort": "low"}},
    )
    for bad_kwargs in bad_cases:
        create_mock.reset_mock()
        with pytest.raises(BadRequestError) as exc_info:
            await client.acompletion(
                f"anthropic/{_MODEL}",
                [{"role": "user", "content": "Hi"}],
                **bad_kwargs,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider == "anthropic"
        create_mock.assert_not_called()
