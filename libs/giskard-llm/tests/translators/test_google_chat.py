"""Google Gemini ``generateContent`` translation tests.

Content shape: https://ai.google.dev/api/generate-content#Content
"""

from typing import Literal

import pytest
from giskard.llm.translators.google_chat import GoogleChatTranslator
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

from .sdk_payload_validation import validate_google_contents
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

_MODEL = "gemini-2.0-flash"


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
    """User and model turns map to ``user`` / ``model`` content entries."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply."),
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
                    }
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
                    }
                },
                {
                    "function_call": {
                        "name": "get_local_time",
                        "args": {"timezone": "Asia/Tokyo"},
                    }
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
                    }
                },
                {
                    "function_call": {
                        "name": "get_local_time",
                        "args": {"timezone": "Asia/Tokyo"},
                    }
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


def test_function_message_raises():
    """FunctionMessage is not supported by the Google translator."""
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        FunctionMessage(name="fn", content="result"),
    ]
    with pytest.raises(ValueError, match="Unsupported message role"):
        GoogleChatTranslator.to_google(_MODEL, messages)
