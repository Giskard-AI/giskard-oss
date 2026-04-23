"""Shared tool definitions and message lists for single- and parallel-tool-call translator tests."""

from typing import cast

from giskard.llm.types import (
    ChatMessageParam,
    ResponseInputItemParam,
    ToolDefParam,
)

WEATHER_TOOL: ToolDefParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

TOOL_CALL_ID = "call_weather_1"

TOOL_RESULT_CONTENT = '{"temperature_c": 22, "conditions": "sunny"}'


def user_assistant_tool_then_tool_result() -> list[ChatMessageParam]:
    """User question, model proposes a function call, tool returns a string result."""
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
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


GET_TIME_TOOL: ToolDefParam = {
    "type": "function",
    "function": {
        "name": "get_local_time",
        "description": "Get local time for an IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone identifier",
                },
            },
            "required": ["timezone"],
        },
    },
}

PARALLEL_TOOLS: list[ToolDefParam] = [WEATHER_TOOL, GET_TIME_TOOL]

TOOL_CALL_ID_WEATHER_PARALLEL = "call_parallel_weather"
TOOL_CALL_ID_TIME_PARALLEL = "call_parallel_time"

TOOL_RESULT_WEATHER_PARALLEL = '{"temperature_c": 18}'
TOOL_RESULT_TIME_PARALLEL = '{"hour": 14, "minute": 30}'

PARALLEL_USER_PROMPT = "What's the weather in Paris and the time in Tokyo?"

ASSISTANT_TEXT_WITH_PARALLEL_TOOLS = "I'll fetch the weather and the local time."


def user_two_parallel_tool_calls_two_results() -> list[ChatMessageParam]:
    """[user, assistant with 2 tool_calls, 2 tool results] — parallel calls, no assistant text."""
    return [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                },
                {
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "type": "function",
                    "function": {
                        "name": "get_local_time",
                        "arguments": {"timezone": "Asia/Tokyo"},
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


def user_message_two_parallel_tool_calls_two_results() -> list[ChatMessageParam]:
    """Same as parallel calls, but the assistant turn also includes visible text."""
    base = user_two_parallel_tool_calls_two_results()
    assistant = base[1]
    if assistant["role"] != "assistant":
        msg = "expected assistant message at index 1"
        raise TypeError(msg)
    tool_calls = assistant.get("tool_calls")
    if tool_calls is None:
        msg = "expected tool_calls on assistant message"
        raise TypeError(msg)
    return [
        base[0],
        {
            "role": "assistant",
            "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            "tool_calls": tool_calls,
        },
        base[2],
        base[3],
    ]


# -- Responses / Interactions API (flat ``ResponseInputItemParam`` lists) --------------


def openai_response_user_tool_call_then_result() -> list[ResponseInputItemParam]:
    """[user, function_call, function_call_output] for OpenAI Responses (no extra keys)."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in Paris?",
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID,
                "output": TOOL_RESULT_CONTENT,
            },
        ],
    )


def google_response_user_tool_call_then_result() -> list[ResponseInputItemParam]:
    """Same conversation as :func:`openai_response_user_tool_call_then_result` with Google-required ids."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in Paris?",
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID,
                "id": TOOL_CALL_ID,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call_output",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID,
                "output": TOOL_RESULT_CONTENT,
            },
        ],
    )


def openai_response_user_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItemParam
]:
    """[user, 2× function_call, 2× function_call_output] (parallel tool calls, no assistant text)."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": PARALLEL_USER_PROMPT,
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "arguments": {"timezone": "Asia/Tokyo"},
            },
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "output": TOOL_RESULT_WEATHER_PARALLEL,
            },
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "output": TOOL_RESULT_TIME_PARALLEL,
            },
        ],
    )


def google_response_user_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItemParam
]:
    """Parallel tool calls and outputs with per-call ``id`` / ``name`` for Gemini Interactions."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": PARALLEL_USER_PROMPT,
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "id": TOOL_CALL_ID_TIME_PARALLEL,
                "arguments": {"timezone": "Asia/Tokyo"},
            },
            {
                "type": "function_call_output",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "output": TOOL_RESULT_WEATHER_PARALLEL,
            },
            {
                "type": "function_call_output",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "output": TOOL_RESULT_TIME_PARALLEL,
            },
        ],
    )


def openai_response_user_assistant_text_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItemParam
]:
    """Assistant text message, then two function calls, then two outputs (parallel with preamble)."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": PARALLEL_USER_PROMPT,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "arguments": {"timezone": "Asia/Tokyo"},
            },
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "output": TOOL_RESULT_WEATHER_PARALLEL,
            },
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "output": TOOL_RESULT_TIME_PARALLEL,
            },
        ],
    )


def google_response_user_assistant_text_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItemParam
]:
    """Same as :func:`openai_response_user_assistant_text_two_parallel_tool_calls_and_results` for Google."""
    return cast(
        list[ResponseInputItemParam],
        [
            {
                "type": "message",
                "role": "user",
                "content": PARALLEL_USER_PROMPT,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "id": TOOL_CALL_ID_TIME_PARALLEL,
                "arguments": {"timezone": "Asia/Tokyo"},
            },
            {
                "type": "function_call_output",
                "name": "get_weather",
                "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
                "output": TOOL_RESULT_WEATHER_PARALLEL,
            },
            {
                "type": "function_call_output",
                "name": "get_local_time",
                "call_id": TOOL_CALL_ID_TIME_PARALLEL,
                "output": TOOL_RESULT_TIME_PARALLEL,
            },
        ],
    )
