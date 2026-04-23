"""OpenAI Responses API translation tests.

Request shape mirrors :meth:`giskard.llm.translators.openai_response.OpenAIResponseTranslator.to_openai`.
For **return** mapping -> :class:`~giskard.llm.types.ResponseResult`, see ``test_openai_response_return.py``.
For **Chat Completions** -> :class:`~giskard.llm.types.CompletionResponse`, see ``test_openai_chat_return.py``.
"""

import json
from typing import Literal

import pytest
from giskard.llm.translators.openai_response import OpenAIResponseTranslator
from giskard.llm.types import (
    ResponseEasyInputMessage,
    ResponseInputItem,
)

from .sdk_payload_validation import validate_openai_response_params
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
    openai_response_user_assistant_text_two_parallel_tool_calls_and_results,
    openai_response_user_tool_call_then_result,
    openai_response_user_two_parallel_tool_calls_and_results,
)

_MODEL = "gpt-4o-mini"


def _message(
    role: Literal["user", "assistant", "system", "developer"],
    content: str,
) -> ResponseInputItem:
    """Easy message items with an explicit ``type`` (mirrors API easy-input messages)."""
    return ResponseEasyInputMessage(
        role=role,
        content=content,
    )


def test_string_input():
    """Plain string input is passed through as ``input`` (typical one-shot prompt)."""
    user_prompt = "Hello."
    payload = OpenAIResponseTranslator.to_openai(_MODEL, user_prompt)

    assert payload.get("model") == _MODEL
    assert payload.get("input") == user_prompt
    assert "instructions" not in payload
    validate_openai_response_params(payload)


def test_string_input_with_instructions():
    """``instructions`` is set separately; user text stays in ``input``."""
    user_prompt = "Hello."
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL,
        user_prompt,
        instructions="You are helpful.",
    )

    assert payload.get("model") == _MODEL
    assert payload.get("input") == user_prompt
    assert payload.get("instructions") == "You are helpful."
    validate_openai_response_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_message_instruction_then_user(
    instruction_role: Literal["system", "developer"],
):
    """List input: system or developer, then user (structured ``input``, like chat)."""
    items: list[ResponseInputItem] = [
        _message(instruction_role, "You are helpful."),
        _message("user", "Hello."),
    ]
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items)

    assert payload.get("input") == [
        {"type": "message", "role": instruction_role, "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hello."},
    ]
    assert "instructions" not in payload
    validate_openai_response_params(payload)


def test_message_system_then_developer_then_user():
    """System and developer are separate list items, then user (like chat)."""
    items: list[ResponseInputItem] = [
        _message("system", "You are helpful."),
        _message("developer", "App version 2.0"),
        _message("user", "Hello."),
    ]
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items)

    assert payload.get("input") == [
        {"type": "message", "role": "system", "content": "You are helpful."},
        {"type": "message", "role": "developer", "content": "App version 2.0"},
        {"type": "message", "role": "user", "content": "Hello."},
    ]
    validate_openai_response_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_message_two_instructions_then_user(
    instruction_role: Literal["system", "developer"],
):
    """Two consecutive system or developer messages, then user (like chat)."""
    items: list[ResponseInputItem]
    if instruction_role == "system":
        items = [
            _message("system", "First system instruction."),
            _message("system", "Second system instruction."),
            _message("user", "Hello."),
        ]
    else:
        items = [
            _message("developer", "First system instruction."),
            _message("developer", "Second system instruction."),
            _message("user", "Hello."),
        ]
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items)

    assert payload.get("input") == [
        {
            "type": "message",
            "role": instruction_role,
            "content": "First system instruction.",
        },
        {
            "type": "message",
            "role": instruction_role,
            "content": "Second system instruction.",
        },
        {"type": "message", "role": "user", "content": "Hello."},
    ]
    validate_openai_response_params(payload)


def test_message_user_assistant_user():
    """Multi-turn: user, assistant, user in ``input`` (like chat)."""
    items: list[ResponseInputItem] = [
        _message("user", "First user."),
        _message("assistant", "Assistant reply."),
        _message("user", "Second user."),
    ]
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items)

    assert payload.get("input") == [
        {"type": "message", "role": "user", "content": "First user."},
        {"type": "message", "role": "assistant", "content": "Assistant reply."},
        {"type": "message", "role": "user", "content": "Second user."},
    ]
    validate_openai_response_params(payload)


def test_user_tool_call_and_result_with_tools():
    """Tool definition plus [user, function_call, function_call_output] (like chat)."""
    items = openai_response_user_tool_call_then_result()
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items, tools=[WEATHER_TOOL])

    assert payload.get("tools") == [
        {
            "type": "function",
            **WEATHER_TOOL.function.model_dump(),
            "strict": None,
        },
    ]
    assert payload.get("input") == [
        {
            "type": "message",
            "role": "user",
            "content": "What's the weather in Paris?",
        },
        {
            "type": "function_call",
            "name": "get_weather",
            "call_id": TOOL_CALL_ID,
            "arguments": json.dumps({"city": "Paris"}),
        },
        {
            "type": "function_call_output",
            "call_id": TOOL_CALL_ID,
            "output": TOOL_RESULT_CONTENT,
        },
    ]
    validate_openai_response_params(payload)


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Two ``function_call`` items, then two ``function_call_output`` items (like chat)."""
    items = openai_response_user_two_parallel_tool_calls_and_results()
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            **WEATHER_TOOL.function.model_dump(),
            "strict": None,
        },
        {
            "type": "function",
            **GET_TIME_TOOL.function.model_dump(),
            "strict": None,
        },
    ]
    assert payload.get("input") == [
        {"type": "message", "role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "type": "function_call",
            "name": "get_weather",
            "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
            "arguments": json.dumps({"city": "Paris"}),
        },
        {
            "type": "function_call",
            "name": "get_local_time",
            "call_id": TOOL_CALL_ID_TIME_PARALLEL,
            "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
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
    ]
    validate_openai_response_params(payload)


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Assistant ``message`` with visible text, then two calls and two outputs (like chat)."""
    items = openai_response_user_assistant_text_two_parallel_tool_calls_and_results()
    payload = OpenAIResponseTranslator.to_openai(_MODEL, items, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            **WEATHER_TOOL.function.model_dump(),
            "strict": None,
        },
        {
            "type": "function",
            **GET_TIME_TOOL.function.model_dump(),
            "strict": None,
        },
    ]
    assert payload.get("input") == [
        {"type": "message", "role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "type": "message",
            "role": "assistant",
            "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
        },
        {
            "type": "function_call",
            "name": "get_weather",
            "call_id": TOOL_CALL_ID_WEATHER_PARALLEL,
            "arguments": json.dumps({"city": "Paris"}),
        },
        {
            "type": "function_call",
            "name": "get_local_time",
            "call_id": TOOL_CALL_ID_TIME_PARALLEL,
            "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
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
    ]
    validate_openai_response_params(payload)
