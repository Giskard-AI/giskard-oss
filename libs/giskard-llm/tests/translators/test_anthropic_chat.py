"""Anthropic Messages API translation tests.

Request shape: https://docs.anthropic.com/en/api/messages
"""

from typing import Literal

import pytest
from giskard.llm.translators.anthropic import AnthropicChatTranslator
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
    """Multi-turn chat: user string content vs assistant text blocks."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply."),
        UserMessage(content="Second user."),
    ]
    payload = AnthropicChatTranslator.to_anthropic(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "user", "content": "First user."},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant reply."}],
        },
        {"role": "user", "content": "Second user."},
    ]
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
