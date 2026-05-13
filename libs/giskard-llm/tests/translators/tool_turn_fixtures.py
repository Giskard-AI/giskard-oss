"""Shared tool definitions and message lists for single- and parallel-tool-call translator tests."""

from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    FunctionDef,
    ResponseEasyInputMessage,
    ResponseFunctionCallOutput,
    ResponseFunctionToolCall,
    ResponseInputItem,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    ToolMessage,
    UserMessage,
)

WEATHER_TOOL = ToolDef(
    function=FunctionDef(
        name="get_weather",
        description="Get weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    ),
)

TOOL_CALL_ID = "call_weather_1"

TOOL_RESULT_CONTENT = '{"temperature_c": 22, "conditions": "sunny"}'


def user_assistant_tool_then_tool_result() -> list[ChatMessage]:
    """User question, model proposes a function call, tool returns a string result."""
    return [
        UserMessage(content="What's the weather in Paris?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=TOOL_CALL_ID,
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments={"city": "Paris"},
                    ),
                )
            ],
        ),
        ToolMessage(
            content=TOOL_RESULT_CONTENT,
            tool_call_id=TOOL_CALL_ID,
        ),
    ]


GET_TIME_TOOL = ToolDef(
    function=FunctionDef(
        name="get_local_time",
        description="Get local time for an IANA timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone identifier",
                },
            },
            "required": ["timezone"],
        },
    ),
)

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
        UserMessage(content=PARALLEL_USER_PROMPT),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=TOOL_CALL_ID_WEATHER_PARALLEL,
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments={"city": "Paris"},
                    ),
                ),
                ToolCall(
                    id=TOOL_CALL_ID_TIME_PARALLEL,
                    function=ToolCallFunction(
                        name="get_local_time",
                        arguments={"timezone": "Asia/Tokyo"},
                    ),
                ),
            ],
        ),
        ToolMessage(
            content=TOOL_RESULT_WEATHER_PARALLEL,
            tool_call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
        ),
        ToolMessage(
            content=TOOL_RESULT_TIME_PARALLEL,
            tool_call_id=TOOL_CALL_ID_TIME_PARALLEL,
        ),
    ]


def user_message_two_parallel_tool_calls_two_results() -> list[ChatMessage]:
    """Same as parallel calls, but the assistant turn also includes visible text."""
    return [
        UserMessage(content=PARALLEL_USER_PROMPT),
        AssistantMessage(
            content=ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            tool_calls=[
                ToolCall(
                    id=TOOL_CALL_ID_WEATHER_PARALLEL,
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments={"city": "Paris"},
                    ),
                ),
                ToolCall(
                    id=TOOL_CALL_ID_TIME_PARALLEL,
                    function=ToolCallFunction(
                        name="get_local_time",
                        arguments={"timezone": "Asia/Tokyo"},
                    ),
                ),
            ],
        ),
        ToolMessage(
            content=TOOL_RESULT_WEATHER_PARALLEL,
            tool_call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
        ),
        ToolMessage(
            content=TOOL_RESULT_TIME_PARALLEL,
            tool_call_id=TOOL_CALL_ID_TIME_PARALLEL,
        ),
    ]


# -- Responses / Interactions API (flat ``ResponseInputItem`` lists) --------------


def openai_response_user_tool_call_then_result() -> list[ResponseInputItem]:
    """[user, function_call, function_call_output] for OpenAI Responses (no extra keys)."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content="What's the weather in Paris?",
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionCallOutput(
            call_id=TOOL_CALL_ID,
            output=TOOL_RESULT_CONTENT,
        ),
    ]


def google_response_user_tool_call_then_result() -> list[ResponseInputItem]:
    """Same conversation as :func:`openai_response_user_tool_call_then_result` with Google-required names on outputs."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content="What's the weather in Paris?",
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionCallOutput(
            name="get_weather",
            call_id=TOOL_CALL_ID,
            output=TOOL_RESULT_CONTENT,
        ),
    ]


def openai_response_user_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItem
]:
    """[user, 2× function_call, 2× function_call_output] (parallel tool calls, no assistant text)."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content=PARALLEL_USER_PROMPT,
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionToolCall(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            arguments={"timezone": "Asia/Tokyo"},
        ),
        ResponseFunctionCallOutput(
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            output=TOOL_RESULT_WEATHER_PARALLEL,
        ),
        ResponseFunctionCallOutput(
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            output=TOOL_RESULT_TIME_PARALLEL,
        ),
    ]


def google_response_user_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItem
]:
    """Parallel tool calls and outputs with ``name`` on each ``function_call_output`` for Gemini Interactions."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content=PARALLEL_USER_PROMPT,
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionToolCall(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            arguments={"timezone": "Asia/Tokyo"},
        ),
        ResponseFunctionCallOutput(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            output=TOOL_RESULT_WEATHER_PARALLEL,
        ),
        ResponseFunctionCallOutput(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            output=TOOL_RESULT_TIME_PARALLEL,
        ),
    ]


def openai_response_user_assistant_text_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItem
]:
    """Assistant text message, then two function calls, then two outputs (parallel with preamble)."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content=PARALLEL_USER_PROMPT,
        ),
        ResponseEasyInputMessage(
            role="assistant",
            content=ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionToolCall(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            arguments={"timezone": "Asia/Tokyo"},
        ),
        ResponseFunctionCallOutput(
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            output=TOOL_RESULT_WEATHER_PARALLEL,
        ),
        ResponseFunctionCallOutput(
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            output=TOOL_RESULT_TIME_PARALLEL,
        ),
    ]


def google_response_user_assistant_text_two_parallel_tool_calls_and_results() -> list[
    ResponseInputItem
]:
    """Same as :func:`openai_response_user_assistant_text_two_parallel_tool_calls_and_results` for Google."""
    return [
        ResponseEasyInputMessage(
            role="user",
            content=PARALLEL_USER_PROMPT,
        ),
        ResponseEasyInputMessage(
            role="assistant",
            content=ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
        ),
        ResponseFunctionToolCall(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            arguments={"city": "Paris"},
        ),
        ResponseFunctionToolCall(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            arguments={"timezone": "Asia/Tokyo"},
        ),
        ResponseFunctionCallOutput(
            name="get_weather",
            call_id=TOOL_CALL_ID_WEATHER_PARALLEL,
            output=TOOL_RESULT_WEATHER_PARALLEL,
        ),
        ResponseFunctionCallOutput(
            name="get_local_time",
            call_id=TOOL_CALL_ID_TIME_PARALLEL,
            output=TOOL_RESULT_TIME_PARALLEL,
        ),
    ]
