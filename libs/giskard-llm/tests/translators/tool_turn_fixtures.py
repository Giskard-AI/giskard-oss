"""Shared tool definitions and message lists for single- and parallel-tool-call translator tests."""

from giskard.llm.types import ChatMessage, ToolDef

WEATHER_TOOL: ToolDef = {
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


def user_assistant_tool_then_tool_result() -> list[ChatMessage]:
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


GET_TIME_TOOL: ToolDef = {
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

PARALLEL_TOOLS: list[ToolDef] = [WEATHER_TOOL, GET_TIME_TOOL]

TOOL_CALL_ID_WEATHER_PARALLEL = "call_parallel_weather"
TOOL_CALL_ID_TIME_PARALLEL = "call_parallel_time"

TOOL_RESULT_WEATHER_PARALLEL = '{"temperature_c": 18}'
TOOL_RESULT_TIME_PARALLEL = '{"hour": 14, "minute": 30}'

PARALLEL_USER_PROMPT = "What's the weather in Paris and the time in Tokyo?"

ASSISTANT_TEXT_WITH_PARALLEL_TOOLS = "I'll fetch the weather and the local time."


def user_two_parallel_tool_calls_two_results() -> list[ChatMessage]:
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


def user_message_two_parallel_tool_calls_two_results() -> list[ChatMessage]:
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
