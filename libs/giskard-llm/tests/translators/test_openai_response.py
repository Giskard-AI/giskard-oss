"""OpenAI Responses API translation tests.

Request shape mirrors :meth:`giskard.llm.translators.openai_response.OpenAIResponseTranslator.to_openai`.
For **return** mapping -> :class:`~giskard.llm.types.ResponseResult`, see ``test_openai_response_return.py``.
For **Chat Completions** -> :class:`~giskard.llm.types.CompletionResponse`, see ``test_openai_chat_return.py``.
"""

import json
import logging
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.routing import LLMClient
from giskard.llm.translators.openai_response import OpenAIResponseTranslator
from giskard.llm.types import (
    ReasoningEffort,
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


def _reasoning_turn(reasoning: object) -> list[ResponseInputItem]:
    from giskard.llm.types import (
        ResponseFunctionCallOutput,
        ResponseFunctionToolCall,
        ResponseResult,
    )

    result = ResponseResult.model_validate({"id": "resp_1", "output": [reasoning]})
    return [
        _message("user", "Weather?"),
        *result.outputs,
        ResponseFunctionToolCall(
            call_id=TOOL_CALL_ID, name="get_weather", arguments={}
        ),
        ResponseFunctionCallOutput(call_id=TOOL_CALL_ID, output=TOOL_RESULT_CONTENT),
    ]


_OPENAI_REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Plan: call the tool."}],
    "content": [{"type": "reasoning_text", "text": "Raw thinking"}],
    "encrypted_content": "enc-blob",
    "signature": "sig",
    "status": "completed",
    "provider": "openai",
}


def test_reasoning_item_round_trips_on_its_route():
    """An ``openai`` reasoning item, stored as JSON and reloaded, replays on ``openai``.

    Only ``type``, ``id``, ``summary`` and ``encrypted_content`` reach the wire:
    never ``content``, ``signature`` or ``provider``.
    """
    from giskard.llm.types import ResponseResult

    stored = ResponseResult.model_validate(
        {"id": "resp_1", "output": [_OPENAI_REASONING]}
    ).model_dump_json()
    reloaded = json.loads(stored)["outputs"][0]

    payload = OpenAIResponseTranslator.to_openai(
        _MODEL, _reasoning_turn(reloaded), provider="openai"
    )
    assert list(payload.get("input", []))[1] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Plan: call the tool."}],
        "encrypted_content": "enc-blob",
    }
    validate_openai_response_params(payload)


@pytest.mark.parametrize(
    ("origin", "route"),
    [
        pytest.param("openai", "azure", id="openai-to-azure"),
        pytest.param("openai", "azure_ai", id="openai-to-azure_ai"),
        pytest.param("azure", "openai", id="azure-to-openai"),
        pytest.param("google", "openai", id="google-to-openai"),
        pytest.param(None, "openai", id="untagged"),
    ],
)
def test_reasoning_item_from_another_route_is_dropped(origin: str | None, route: str):
    """Reasoning replays only on its exact Originating Provider; ``None`` never matches."""
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL,
        _reasoning_turn({**_OPENAI_REASONING, "provider": origin}),
        provider=route,
    )
    items = list(payload.get("input", []))
    assert [cast(dict[str, Any], cast(object, i)).get("type") for i in items] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    validate_openai_response_params(payload)


def test_openai_reasoning_item_without_id_is_rejected():
    """OpenAI needs the item ``id`` to replay reasoning; it is never invented."""
    from giskard.llm.types import ReasoningContent

    with pytest.raises(ValueError, match="id"):
        OpenAIResponseTranslator.to_openai(
            _MODEL,
            [
                _message("user", "Hi"),
                ReasoningContent(encrypted_content="enc", provider="openai"),
            ],
            provider="openai",
        )


async def test_azure_client_drops_openai_reasoning(monkeypatch: pytest.MonkeyPatch):
    """``LLMClient.aresponse`` on ``azure`` does not replay ``openai`` reasoning."""
    monkeypatch.setenv("AZURE_API_KEY", "k")
    monkeypatch.setenv("AZURE_API_BASE", "https://x.openai.azure.com")
    client = LLMClient()
    provider = client._get_provider("azure")
    sdk_resp = MagicMock()
    sdk_resp.model_dump.return_value = {"id": "resp_2", "output": [], "model": "m"}
    create_mock = AsyncMock(return_value=sdk_resp)
    cast(Any, provider)._client = MagicMock(responses=MagicMock(create=create_mock))

    await client.aresponse(
        "azure/gpt-5", cast(Any, [{"role": "user", "content": "Hi"}, _OPENAI_REASONING])
    )

    sent = create_mock.call_args.kwargs["input"]
    assert [i.get("type") for i in sent] == ["message"]


@pytest.mark.parametrize(
    ("raw_effort", "expected"),
    [
        pytest.param("none", "none", id="none"),
        pytest.param("minimal", "minimal", id="minimal"),
        pytest.param("low", "low", id="low"),
        pytest.param("medium", "medium", id="medium"),
        pytest.param("high", "high", id="high"),
        pytest.param("xhigh", "xhigh", id="xhigh"),
        pytest.param("max", "max", id="max"),
        pytest.param("HIGH", "high", id="upper-case"),
        pytest.param("xHiGh", "xhigh", id="mixed-case"),
    ],
)
def test_reasoning_effort_canonical_levels_map_to_reasoning_dict(
    raw_effort: str, expected: ReasoningEffort, caplog: pytest.LogCaptureFixture
):
    """All 7 canonical levels map to ``reasoning={"effort": ...}`` without top-level leak."""
    with caplog.at_level(
        logging.WARNING, logger="giskard.llm.translators.openai_response"
    ):
        payload = OpenAIResponseTranslator.to_openai(
            _MODEL, "Hello.", reasoning_effort=raw_effort
        )
    assert "reasoning_effort" not in payload
    assert payload.get("reasoning") == {"effort": expected}
    assert "reasoning_effort" not in " ".join(r.message for r in caplog.records)
    validate_openai_response_params(payload)


def test_reasoning_effort_none_omitted():
    """Explicit ``reasoning_effort=None`` omits ``reasoning`` from the payload."""
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL, "Hello.", reasoning_effort=None
    )
    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload
    validate_openai_response_params(payload)


def test_reasoning_escape_hatch_merges_without_overwriting_siblings(
    caplog: pytest.LogCaptureFixture,
):
    """``reasoning_effort`` merges into ``reasoning`` preserving sibling keys."""
    escape = {"summary": "concise", "context": "current_turn"}
    with caplog.at_level(
        logging.WARNING, logger="giskard.llm.translators.openai_response"
    ):
        payload = OpenAIResponseTranslator.to_openai(
            _MODEL, "Hello.", reasoning_effort="HIGH", reasoning=escape
        )
    assert payload.get("reasoning") == {
        "summary": "concise",
        "context": "current_turn",
        "effort": "high",
    }
    assert escape == {"summary": "concise", "context": "current_turn"}
    assert "reasoning" not in " ".join(r.message for r in caplog.records)
    validate_openai_response_params(payload)


def test_reasoning_escape_hatch_only_passes_through():
    """Explicit ``reasoning`` without ``reasoning_effort`` is forwarded verbatim."""
    payload = OpenAIResponseTranslator.to_openai(
        _MODEL, "Hello.", reasoning={"summary": "detailed"}
    )
    assert payload.get("reasoning") == {"summary": "detailed"}
    validate_openai_response_params(payload)


def test_reasoning_effort_and_reasoning_effort_key_conflict_raises():
    """Providing both ``reasoning_effort`` and ``reasoning['effort']`` raises 400."""
    with pytest.raises(BadRequestError) as exc_info:
        OpenAIResponseTranslator.to_openai(
            _MODEL,
            "Hello.",
            reasoning_effort="high",
            reasoning={"effort": "low", "summary": "concise"},
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.provider == "openai"


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("openai/gpt-5", id="openai"),
        pytest.param("azure/gpt-5", id="azure"),
        pytest.param("azure_ai/gpt-5", id="azure_ai"),
    ],
)
async def test_client_aresponse_forwards_and_rejects_reasoning_effort(
    model_id: str, monkeypatch: pytest.MonkeyPatch
):
    """``LLMClient.aresponse`` forwards merged reasoning and rejects invalid/conflict before SDK."""
    if model_id.startswith("azure_ai/"):
        monkeypatch.setenv("AZURE_AI_API_KEY", "k")
        monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://x.services.ai.azure.com")
    elif model_id.startswith("azure/"):
        monkeypatch.setenv("AZURE_API_KEY", "k")
        monkeypatch.setenv("AZURE_API_BASE", "https://x.openai.azure.com")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "k")

    client = LLMClient()
    provider = client._get_provider(model_id.split("/")[0])
    sdk_resp = MagicMock()
    sdk_resp.model_dump.return_value = {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Ok"}],
            }
        ],
        "model": "gpt-5",
    }
    create_mock = AsyncMock(return_value=sdk_resp)
    cast(Any, provider)._client = MagicMock(responses=MagicMock(create=create_mock))

    await client.aresponse(
        model_id,
        "Hi",
        reasoning_effort="HIGH",
        reasoning={"summary": "auto"},
    )
    assert create_mock.call_args.kwargs["reasoning"] == {
        "summary": "auto",
        "effort": "high",
    }

    bad_cases: tuple[dict[str, Any], ...] = (
        {"reasoning_effort": "ultra"},
        {"reasoning_effort": "high", "reasoning": {"effort": "low"}},
    )
    for bad_kwargs in bad_cases:
        create_mock.reset_mock()
        with pytest.raises(BadRequestError) as exc_info:
            await client.aresponse(model_id, "Hi", **bad_kwargs)
        assert exc_info.value.status_code == 400
        create_mock.assert_not_called()
