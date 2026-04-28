"""OpenAI Chat Completions translation tests.

Request shape: https://platform.openai.com/docs/api-reference/chat/create
"""

import json
from typing import Literal, cast

import pytest
from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    DeveloperMessage,
    FunctionMessage,
    RefusalContent,
    SystemMessage,
    TextContent,
    UserMessage,
)
from pydantic import BaseModel

from .sdk_payload_validation import validate_openai_completion_params
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

_MODEL = "gpt-4o-mini"


def test_single_user_message():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(content="Hello.")
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    validate_openai_completion_params(payload)


def test_single_user_message_with_text_content():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(content=[TextContent(text="Hello.")])
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hello."}]}
    ]
    validate_openai_completion_params(payload)


def test_single_user_message_with_text_contents():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(
        content=[TextContent(text="Hello."), TextContent(text="World.")]
    )
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello."},
                {"type": "text", "text": "World."},
            ],
        }
    ]
    validate_openai_completion_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_instruction_then_user(instruction_role: Literal["system", "developer"]):
    """System or developer prompt precedes the first user message (distinct roles for OpenAI)."""
    first: SystemMessage | DeveloperMessage = (
        SystemMessage(content="You are helpful.")
        if instruction_role == "system"
        else DeveloperMessage(content="You are helpful.")
    )
    messages: list[ChatMessage] = [
        first,
        UserMessage(content="Hello."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": instruction_role, "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


def test_system_then_developer_then_user():
    """System and developer are separate turns for OpenAI."""
    messages: list[ChatMessage] = [
        SystemMessage(content="You are helpful."),
        DeveloperMessage(content="App version 2.0"),
        UserMessage(content="Hello."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "developer", "content": "App version 2.0"},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_two_instructions_then_user(instruction_role: Literal["system", "developer"]):
    """Two consecutive system or developer turns stay separate in Chat Completions."""
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
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": instruction_role, "content": "First system instruction."},
        {"role": instruction_role, "content": "Second system instruction."},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


def test_user_assistant_user():
    """Multi-turn chat preserves alternating user / assistant / user."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply."),
        UserMessage(content="Second user."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "user", "content": "First user."},
        {"role": "assistant", "content": "Assistant reply."},
        {"role": "user", "content": "Second user."},
    ]
    validate_openai_completion_params(payload)


def test_assistant_refusal_string():
    """Assistant turn may replay OpenAI ``refusal`` without normal ``content``."""
    messages: list[ChatMessage] = [
        UserMessage(content="Do something unsafe."),
        AssistantMessage(refusal="I'm sorry, I can't help with that."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {"role": "user", "content": "Do something unsafe."},
        {
            "role": "assistant",
            "refusal": "I'm sorry, I can't help with that.",
        },
    ]
    validate_openai_completion_params(payload)


def test_assistant_mixed_text_and_refusal_content_parts():
    """Structured assistant ``content`` with text and refusal parts maps to OpenAI parts."""
    messages: list[ChatMessage] = [
        AssistantMessage(
            content=[
                TextContent(text="Partial."),
                RefusalContent(refusal="Declined."),
            ],
        ),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Partial."},
                {"type": "refusal", "refusal": "Declined."},
            ],
        },
    ]
    validate_openai_completion_params(payload)


def test_user_tool_call_and_result_with_tools():
    """Tool definition plus [user, assistant w/ tool_calls, tool result] for Chat Completions."""
    messages = user_assistant_tool_then_tool_result()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=[WEATHER_TOOL])

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": TOOL_RESULT_CONTENT,
            "tool_call_id": TOOL_CALL_ID,
        },
    ]
    validate_openai_completion_params(payload)


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Parallel tool_use: two ``tool_calls`` on one assistant turn, two ``tool`` results."""
    messages = user_two_parallel_tool_calls_two_results()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "Get local time for an IANA timezone.",
                "parameters": GET_TIME_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                },
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "function": {
                        "name": "get_local_time",
                        "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": TOOL_RESULT_WEATHER_PARALLEL,
            "tool_call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
        },
        {
            "role": "tool",
            "content": TOOL_RESULT_TIME_PARALLEL,
            "tool_call_id": TOOL_CALL_ID_TIME_PARALLEL,
        },
    ]
    validate_openai_completion_params(payload)


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Assistant turn with both user-visible text and two parallel ``tool_calls``."""
    messages = user_message_two_parallel_tool_calls_two_results()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "Get local time for an IANA timezone.",
                "parameters": GET_TIME_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                },
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "function": {
                        "name": "get_local_time",
                        "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": TOOL_RESULT_WEATHER_PARALLEL,
            "tool_call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
        },
        {
            "role": "tool",
            "content": TOOL_RESULT_TIME_PARALLEL,
            "tool_call_id": TOOL_CALL_ID_TIME_PARALLEL,
        },
    ]
    validate_openai_completion_params(payload)


def test_function_message_passes_through():
    """FunctionMessage maps to the legacy OpenAI function-role wire format."""
    messages: list[ChatMessage] = [
        UserMessage(content="What is 2+2?"),
        FunctionMessage(name="calculate", content="4"),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "function", "name": "calculate", "content": "4"},
    ]


class _OptionalReasonModel(BaseModel):
    passed: bool
    reason: str | None = None


def test_response_format_pydantic_model_json_schema_without_strict():
    """Coerce ``response_format`` BaseModel to json_schema without OpenAI strict mode.

    Strict structured output rejects Pydantic schemas where some properties are not
    listed in ``required`` (e.g. optional nullable ``reason``); non-strict json_schema
    must still validate as Chat Completions params.
    """
    msg: UserMessage = UserMessage(content="Hi.")
    payload_raw = OpenAIChatTranslator.to_openai(
        _MODEL, [msg], response_format=_OptionalReasonModel
    )
    payload = json.loads(json.dumps(cast(object, payload_raw)))
    rf = payload["response_format"]
    assert isinstance(rf, dict)
    assert rf["type"] == "json_schema"
    inner = rf["json_schema"]
    assert isinstance(inner, dict)
    assert inner["name"] == "_OptionalReasonModel"
    assert "strict" not in inner
    schema_inner = inner["schema"]
    assert isinstance(schema_inner, dict)
    assert schema_inner.get("additionalProperties") is False
    validate_openai_completion_params(payload_raw)
