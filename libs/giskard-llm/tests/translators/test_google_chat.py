"""Google Gemini ``generateContent`` translation tests.

Content shape: https://ai.google.dev/api/generate-content#Content
"""

from giskard.llm.translators.google_chat import GoogleChatTranslator
from giskard.llm.types import ChatMessage, UserMessage

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
    msg: UserMessage = {"role": "user", "content": "Hello."}
    payload = GoogleChatTranslator.to_google(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert payload.get("config", {}) == {}
    validate_google_contents(payload["contents"])


def test_system_then_user():
    """System prompts become ``system_instruction``; user text stays in ``contents``."""
    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
    assert "config" in payload
    cfg = payload["config"]
    assert cfg.get("system_instruction") == ["You are helpful."]
    validate_google_contents(payload["contents"])


def test_two_system_then_user():
    """Several system messages are concatenated in order in ``system_instruction``."""
    messages: list[ChatMessage] = [
        {"role": "system", "content": "First system instruction."},
        {"role": "system", "content": "Second system instruction."},
        {"role": "user", "content": "Hello."},
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
        {"role": "user", "content": "First user."},
        {"role": "assistant", "content": "Assistant reply."},
        {"role": "user", "content": "Second user."},
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, messages)

    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "First user."}]},
        {"role": "model", "parts": [{"text": "Assistant reply."}]},
        {"role": "user", "parts": [{"text": "Second user."}]},
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
                    "parameters": WEATHER_TOOL["function"]["parameters"],
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
                    "parameters": WEATHER_TOOL["function"]["parameters"],
                }
            ],
        },
        {
            "function_declarations": [
                {
                    "name": "get_local_time",
                    "description": "Get local time for an IANA timezone.",
                    "parameters": GET_TIME_TOOL["function"]["parameters"],
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
                    "parameters": WEATHER_TOOL["function"]["parameters"],
                }
            ],
        },
        {
            "function_declarations": [
                {
                    "name": "get_local_time",
                    "description": "Get local time for an IANA timezone.",
                    "parameters": GET_TIME_TOOL["function"]["parameters"],
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
