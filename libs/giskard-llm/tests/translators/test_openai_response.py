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


def test_response_format_pydantic_maps_to_text_format():
    from typing import Any, cast

    from .nested_schema_models import NestedOutputModel

    payload = OpenAIResponseTranslator.to_openai(
        _MODEL, "Hello.", response_format=NestedOutputModel
    )
    text = cast(dict[str, Any], cast(object, payload.get("text")))
    schema = text["format"]["schema"]
    assert text["format"]["type"] == "json_schema"
    assert text["format"]["name"] == "NestedOutputModel"
    assert schema["properties"]["inner"]["$ref"] == "#/$defs/NestedInnerModel"
    assert "value" in schema["$defs"]["NestedInnerModel"]["properties"]
    assert schema["additionalProperties"] is False
    assert "response_format" not in payload
    validate_openai_response_params(payload)


def test_response_format_chat_json_schema_dict_maps_to_text_format():
    from typing import Any, cast

    inner_def = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/NestedInnerModel"},
        },
        "required": ["inner"],
        "$defs": {"NestedInnerModel": inner_def},
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "NestedDict", "schema": schema},
    }
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL, "Hello.", response_format=response_format
    )
    text = cast(dict[str, Any], cast(object, payload.get("text")))
    out_schema = text["format"]["schema"]
    assert out_schema["properties"]["inner"]["$ref"] == "#/$defs/NestedInnerModel"
    assert "value" in out_schema["$defs"]["NestedInnerModel"]["properties"]
    assert out_schema["additionalProperties"] is False
    assert "response_format" not in payload
    validate_openai_response_params(payload)


def test_reasoning_item_round_trips_unchanged():
    """A returned ``reasoning`` item is sent back verbatim (``id``, ``encrypted_content``)."""
    from giskard.llm.types import ResponseFunctionCallOutput, ResponseResult

    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Plan: call the tool."}],
        "encrypted_content": "enc-blob",
    }
    result = ResponseResult.model_validate(
        {
            "id": "resp_1",
            "output": [
                reasoning,
                {
                    "type": "function_call",
                    "call_id": TOOL_CALL_ID,
                    "name": "get_weather",
                    "arguments": "{}",
                },
            ],
        }
    )
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL,
        [
            _message("user", "Weather?"),
            *result.outputs,
            ResponseFunctionCallOutput(
                call_id=TOOL_CALL_ID, output=TOOL_RESULT_CONTENT
            ),
        ],
    )
    assert list(payload.get("input", []))[1] == reasoning
    validate_openai_response_params(payload)
