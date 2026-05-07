"""Google Gemini Interactions (response API) translation tests.

Request shape mirrors :meth:`giskard.llm.translators.google_response.GoogleResponseTranslator.to_google`.
Each :class:`~giskard.llm.types.ResponseEasyInputMessage` and message item serializes to one
``TurnParam`` (``content`` + ``role``); :class:`~giskard.llm.types.ResponseFunctionToolCall`
matches assistant replay as a model turn whose ``content`` contains ``function_call`` (``id``, not top-level ``call_id``);
:class:`~giskard.llm.types.ResponseFunctionCallOutput` becomes a **user** turn whose ``content`` contains ``function_result``.
``None`` from system/developer turns is dropped.

**Role mapping** (Gemini Interactions ``TurnParam.role``): ``user`` stays ``user``;
``assistant`` becomes ``model`` (same for :class:`~giskard.llm.types.ResponseOutputMessage`).
:class:`~giskard.llm.types.ResponseOutputFunctionCall` is always ``model``.
System and developer text is not emitted as turns: it is merged into ``system_instruction``.

For **return** mapping -> :class:`~giskard.llm.types.ResponseResult`, see ``test_google_response_return.py``.
For **generateContent** -> :class:`~giskard.llm.types.CompletionResponse`, see ``test_google_chat_return.py``.
"""

from typing import Literal

import pytest
from giskard.llm.translators.google_response import GoogleResponseTranslator
from giskard.llm.types import (
    ResponseEasyInputMessage,
    ResponseInputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from .sdk_payload_validation import validate_google_interaction_params
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
    google_response_user_assistant_text_two_parallel_tool_calls_and_results,
    google_response_user_tool_call_then_result,
    google_response_user_two_parallel_tool_calls_and_results,
)

_MODEL = "gemini-2.0-flash"


def _message(
    role: Literal["user", "assistant", "system", "developer"],
    content: str,
) -> ResponseInputItem:
    """Easy message items with an explicit ``type`` (so system text is not mixed with user)."""
    return ResponseEasyInputMessage(
        role=role,
        content=content,
    )


def test_string_input():
    """Plain string input is passed through as ``input`` (typical one-shot prompt)."""
    user_prompt = "Hello."
    payload = GoogleResponseTranslator.to_google(_MODEL, user_prompt)

    assert payload["model"] == _MODEL
    assert payload["input"] == user_prompt
    assert "system_instruction" not in payload
    validate_google_interaction_params(payload)


def test_string_input_with_instructions():
    """``instructions`` becomes ``system_instruction``; user text stays in ``input``."""
    user_prompt = "Hello."
    payload = GoogleResponseTranslator.to_google(
        _MODEL,
        user_prompt,
        instructions="You are helpful.",
    )

    assert payload["model"] == _MODEL
    assert payload["input"] == user_prompt
    assert payload.get("system_instruction") == "You are helpful."
    validate_google_interaction_params(payload)


_TEXT = {"type": "text"}


def _text_part(text: str) -> dict[str, str]:
    return {**_TEXT, "text": text}


def _msg_turn(role: Literal["user", "model"], text: str) -> dict[str, object]:
    """One ``ResponseEasyInputMessage`` serializes to a single Interactions turn dict."""
    return {"content": [_text_part(text)], "role": role}


def _model_function_call_turn(
    call_id: str, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    """Input ``ResponseFunctionToolCall`` matches assistant replay shape (Interactions API)."""
    return {
        "content": [
            {
                "type": "function_call",
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
        ],
        "role": "model",
    }


def _user_function_result_turn(
    call_id: str, name: str, result: str
) -> dict[str, object]:
    """``ResponseFunctionCallOutput`` → user turn with ``function_result`` content (Gemini API)."""
    return {
        "content": [
            {
                "type": "function_result",
                "call_id": call_id,
                "name": name,
                "result": result,
            }
        ],
        "role": "user",
    }


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_message_instruction_then_user(
    instruction_role: Literal["system", "developer"],
):
    """System or developer is folded to ``system_instruction``; only user in ``input`` (like chat)."""
    items: list[ResponseInputItem] = [
        _message(instruction_role, "You are helpful."),
        _message("user", "Hello."),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [_msg_turn("user", "Hello.")]
    assert payload.get("system_instruction") == "You are helpful."
    validate_google_interaction_params(payload)


def test_message_system_then_developer_then_user():
    """System and developer concatenate in order in ``system_instruction``; user in ``input``."""
    items: list[ResponseInputItem] = [
        _message("system", "You are helpful."),
        _message("developer", "App version 2.0"),
        _message("user", "Hello."),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [_msg_turn("user", "Hello.")]
    assert payload.get("system_instruction") == "You are helpful.\nApp version 2.0"
    validate_google_interaction_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_message_two_instructions_then_user(
    instruction_role: Literal["system", "developer"],
):
    """Two system or developer lines join ``system_instruction``; one user text in ``input``."""
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
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [_msg_turn("user", "Hello.")]
    assert (
        payload.get("system_instruction")
        == "First system instruction.\nSecond system instruction."
    )
    validate_google_interaction_params(payload)


def test_message_user_assistant_user():
    """User and assistant turns map to a flat list of turn dicts in ``input``."""
    items: list[ResponseInputItem] = [
        _message("user", "First user."),
        _message("assistant", "Assistant reply."),
        _message("user", "Second user."),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [
        _msg_turn("user", "First user."),
        _msg_turn("model", "Assistant reply."),
        _msg_turn("user", "Second user."),
    ]
    assert "system_instruction" not in payload
    validate_google_interaction_params(payload)


def test_user_tool_call_and_result_with_tools():
    """Tool declaration plus user text, model turn ``function_call``, then user ``function_result``."""
    items = google_response_user_tool_call_then_result()
    payload = GoogleResponseTranslator.to_google(_MODEL, items, tools=[WEATHER_TOOL])

    assert payload.get("tools") == [
        {"type": "function", **WEATHER_TOOL.function.model_dump()},
    ]
    assert payload.get("input") == [
        _msg_turn("user", "What's the weather in Paris?"),
        _model_function_call_turn(TOOL_CALL_ID, "get_weather", {"city": "Paris"}),
        _user_function_result_turn(TOOL_CALL_ID, "get_weather", TOOL_RESULT_CONTENT),
    ]
    validate_google_interaction_params(payload)


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Two model-turn ``function_call`` entries, then two user turns with ``function_result`` each."""
    items = google_response_user_two_parallel_tool_calls_and_results()
    payload = GoogleResponseTranslator.to_google(_MODEL, items, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {"type": "function", **WEATHER_TOOL.function.model_dump()},
        {"type": "function", **GET_TIME_TOOL.function.model_dump()},
    ]
    assert payload.get("input") == [
        _msg_turn("user", PARALLEL_USER_PROMPT),
        _model_function_call_turn(
            TOOL_CALL_ID_WEATHER_PARALLEL,
            "get_weather",
            {"city": "Paris"},
        ),
        _model_function_call_turn(
            TOOL_CALL_ID_TIME_PARALLEL,
            "get_local_time",
            {"timezone": "Asia/Tokyo"},
        ),
        _user_function_result_turn(
            TOOL_CALL_ID_WEATHER_PARALLEL,
            "get_weather",
            TOOL_RESULT_WEATHER_PARALLEL,
        ),
        _user_function_result_turn(
            TOOL_CALL_ID_TIME_PARALLEL,
            "get_local_time",
            TOOL_RESULT_TIME_PARALLEL,
        ),
    ]
    validate_google_interaction_params(payload)


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Assistant text turn, two model-turn ``function_call`` entries, then two ``function_result`` user turns."""
    items = google_response_user_assistant_text_two_parallel_tool_calls_and_results()
    payload = GoogleResponseTranslator.to_google(_MODEL, items, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {"type": "function", **WEATHER_TOOL.function.model_dump()},
        {"type": "function", **GET_TIME_TOOL.function.model_dump()},
    ]
    assert payload.get("input") == [
        _msg_turn("user", PARALLEL_USER_PROMPT),
        _msg_turn("model", ASSISTANT_TEXT_WITH_PARALLEL_TOOLS),
        _model_function_call_turn(
            TOOL_CALL_ID_WEATHER_PARALLEL,
            "get_weather",
            {"city": "Paris"},
        ),
        _model_function_call_turn(
            TOOL_CALL_ID_TIME_PARALLEL,
            "get_local_time",
            {"timezone": "Asia/Tokyo"},
        ),
        _user_function_result_turn(
            TOOL_CALL_ID_WEATHER_PARALLEL,
            "get_weather",
            TOOL_RESULT_WEATHER_PARALLEL,
        ),
        _user_function_result_turn(
            TOOL_CALL_ID_TIME_PARALLEL,
            "get_local_time",
            TOOL_RESULT_TIME_PARALLEL,
        ),
    ]
    validate_google_interaction_params(payload)


def test_assistant_message_mixed_output_text_and_refusal_maps_to_text_parts():
    """Structured assistant content with text + refusal maps to plain Gemini ``text`` parts."""
    items: list[ResponseInputItem] = [
        ResponseOutputMessage(
            role="assistant",
            content=[
                ResponseOutputText(text="Partial."),
                ResponseOutputRefusal(refusal="Stopped."),
            ],
        ),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)
    assert payload["input"] == [
        {
            "content": [
                {"type": "text", "text": "Partial."},
                {"type": "text", "text": "Stopped."},
            ],
            "role": "model",
        }
    ]
    validate_google_interaction_params(payload)
