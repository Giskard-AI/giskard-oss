"""OpenAI Chat Completions translation tests.

Request shape: https://platform.openai.com/docs/api-reference/chat/create
"""

import json
import logging
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from giskard.llm.errors import BadRequestError
from giskard.llm.routing import LLMClient
from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    DeveloperMessage,
    FunctionMessage,
    ReasoningEffort,
    RefusalContent,
    SystemMessage,
    TextContent,
    UserMessage,
)
from pydantic import BaseModel

from .sdk_payload_validation import validate_openai_completion_params
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

_MODEL = "gpt-4o-mini"


def test_single_user_message():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(content="Hello.")
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [{"role": "user", "content": "Hello."}]
    validate_openai_completion_params(payload)


def test_single_user_message_with_text_content():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(content=[TextContent(text="Hello.")])
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hello."}]}
    ]
    validate_openai_completion_params(payload)


def test_single_user_message_with_text_contents():
    """A lone user turn maps to one ``user`` message (typical conversation start)."""
    msg: UserMessage = UserMessage(
        content=[TextContent(text="Hello."), TextContent(text="World.")]
    )
    payload = OpenAIChatTranslator.to_openai(_MODEL, [msg])

    assert payload["model"] == _MODEL
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello."},
                {"type": "text", "text": "World."},
            ],
        }
    ]
    validate_openai_completion_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_instruction_then_user(instruction_role: Literal["system", "developer"]):
    """System or developer prompt precedes the first user message (distinct roles for OpenAI)."""
    first: SystemMessage | DeveloperMessage = (
        SystemMessage(content="You are helpful.")
        if instruction_role == "system"
        else DeveloperMessage(content="You are helpful.")
    )
    messages: list[ChatMessage] = [
        first,
        UserMessage(content="Hello."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": instruction_role, "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


def test_system_then_developer_then_user():
    """System and developer are separate turns for OpenAI."""
    messages: list[ChatMessage] = [
        SystemMessage(content="You are helpful."),
        DeveloperMessage(content="App version 2.0"),
        UserMessage(content="Hello."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "developer", "content": "App version 2.0"},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_two_instructions_then_user(instruction_role: Literal["system", "developer"]):
    """Two consecutive system or developer turns stay separate in Chat Completions."""
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
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": instruction_role, "content": "First system instruction."},
        {"role": instruction_role, "content": "Second system instruction."},
        {"role": "user", "content": "Hello."},
    ]
    validate_openai_completion_params(payload)


def test_user_assistant_user():
    """Multi-turn chat preserves alternating user / assistant / user."""
    messages: list[ChatMessage] = [
        UserMessage(content="First user."),
        AssistantMessage(content="Assistant reply."),
        UserMessage(content="Second user."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)

    assert payload["messages"] == [
        {"role": "user", "content": "First user."},
        {"role": "assistant", "content": "Assistant reply."},
        {"role": "user", "content": "Second user."},
    ]
    validate_openai_completion_params(payload)


def test_assistant_refusal_string():
    """Assistant turn may replay OpenAI ``refusal`` without normal ``content``."""
    messages: list[ChatMessage] = [
        UserMessage(content="Do something unsafe."),
        AssistantMessage(refusal="I'm sorry, I can't help with that."),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {"role": "user", "content": "Do something unsafe."},
        {
            "role": "assistant",
            "refusal": "I'm sorry, I can't help with that.",
        },
    ]
    validate_openai_completion_params(payload)


def test_assistant_mixed_text_and_refusal_content_parts():
    """Structured assistant ``content`` with text and refusal parts maps to OpenAI parts."""
    messages: list[ChatMessage] = [
        AssistantMessage(
            content=[
                TextContent(text="Partial."),
                RefusalContent(refusal="Declined."),
            ],
        ),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Partial."},
                {"type": "refusal", "refusal": "Declined."},
            ],
        },
    ]
    validate_openai_completion_params(payload)


def test_user_tool_call_and_result_with_tools():
    """Tool definition plus [user, assistant w/ tool_calls, tool result] for Chat Completions."""
    messages = user_assistant_tool_then_tool_result()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=[WEATHER_TOOL])

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
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
    validate_openai_completion_params(payload)


def test_user_two_parallel_tool_calls_and_results_with_tools():
    """Parallel tool_use: two ``tool_calls`` on one assistant turn, two ``tool`` results."""
    messages = user_two_parallel_tool_calls_two_results()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "Get local time for an IANA timezone.",
                "parameters": GET_TIME_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                },
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "function": {
                        "name": "get_local_time",
                        "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
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
    validate_openai_completion_params(payload)


def test_user_assistant_text_two_parallel_tool_calls_and_results_with_tools():
    """Assistant turn with both user-visible text and two parallel ``tool_calls``."""
    messages = user_message_two_parallel_tool_calls_two_results()
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages, tools=PARALLEL_TOOLS)

    assert payload.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": WEATHER_TOOL.function.parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "Get local time for an IANA timezone.",
                "parameters": GET_TIME_TOOL.function.parameters,
            },
        },
    ]
    assert payload["messages"] == [
        {"role": "user", "content": PARALLEL_USER_PROMPT},
        {
            "role": "assistant",
            "content": ASSISTANT_TEXT_WITH_PARALLEL_TOOLS,
            "tool_calls": [
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_WEATHER_PARALLEL,
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                },
                {
                    "type": "function",
                    "id": TOOL_CALL_ID_TIME_PARALLEL,
                    "function": {
                        "name": "get_local_time",
                        "arguments": json.dumps({"timezone": "Asia/Tokyo"}),
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
    validate_openai_completion_params(payload)


def test_function_message_passes_through():
    """FunctionMessage maps to the legacy OpenAI function-role wire format."""
    messages: list[ChatMessage] = [
        UserMessage(content="What is 2+2?"),
        FunctionMessage(name="calculate", content="4"),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assert payload["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "function", "name": "calculate", "content": "4"},
    ]


class _OptionalReasonModel(BaseModel):
    passed: bool
    reason: str | None = None


def test_response_format_pydantic_model_json_schema_without_strict():
    """Coerce ``response_format`` BaseModel to json_schema without OpenAI strict mode.

    Strict structured output rejects Pydantic schemas where some properties are not
    listed in ``required`` (e.g. optional nullable ``reason``); non-strict json_schema
    must still validate as Chat Completions params.
    """
    msg: UserMessage = UserMessage(content="Hi.")
    payload_raw = OpenAIChatTranslator.to_openai(
        _MODEL, [msg], response_format=_OptionalReasonModel
    )
    payload = json.loads(json.dumps(cast(object, payload_raw)))
    rf = payload["response_format"]
    assert isinstance(rf, dict)
    assert rf["type"] == "json_schema"
    inner = rf["json_schema"]
    assert isinstance(inner, dict)
    assert inner["name"] == "_OptionalReasonModel"
    assert "strict" not in inner
    schema_inner = inner["schema"]
    assert isinstance(schema_inner, dict)
    assert schema_inner.get("additionalProperties") is False
    validate_openai_completion_params(payload_raw)


def test_response_format_nested_pydantic_schema_has_additional_properties_on_defs():
    from .nested_schema_models import NestedOutputModel

    msg = UserMessage(content="Hi.")
    payload_raw = OpenAIChatTranslator.to_openai(
        _MODEL, [msg], response_format=NestedOutputModel
    )
    payload = json.loads(json.dumps(cast(object, payload_raw)))
    schema = cast(dict[str, Any], cast(object, payload))["response_format"][
        "json_schema"
    ]["schema"]
    assert schema["additionalProperties"] is False
    validate_openai_completion_params(payload_raw)


def test_response_format_map_keeps_value_schema():
    from pydantic import BaseModel

    class MapOutput(BaseModel):
        values: dict[str, int]

    payload = OpenAIChatTranslator.to_openai(
        _MODEL, [UserMessage(content="Hi.")], response_format=MapOutput
    )
    schema = cast(dict[str, Any], cast(object, payload))["response_format"][
        "json_schema"
    ]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["values"]["additionalProperties"] == {"type": "integer"}


def test_assistant_reasoning_is_stripped_for_official_api():
    """``reasoning`` / ``reasoning_details`` are not official Chat Completions fields: never sent."""
    from giskard.llm.types import ReasoningTextDetail, ToolCall, ToolCallFunction

    messages: list[ChatMessage] = [
        UserMessage(content="What is 6*7?"),
        AssistantMessage(
            content=None,
            reasoning="Multiply.",
            reasoning_details=[ReasoningTextDetail(text="Multiply.", signature="s")],
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="mul", arguments={"a": 6, "b": 7}),
                )
            ],
        ),
    ]
    payload = OpenAIChatTranslator.to_openai(_MODEL, messages)
    assistant = cast(dict[str, Any], cast(object, list(payload["messages"])[1]))
    assert "reasoning" not in assistant
    assert "reasoning_details" not in assistant
    assert assistant["tool_calls"][0]["id"] == "call_1"
    # The caller's message is left untouched.
    assert cast(AssistantMessage, messages[1]).reasoning == "Multiply."
    validate_openai_completion_params(payload)


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
def test_reasoning_effort_canonical_levels_emitted(
    raw_effort: str, expected: ReasoningEffort, caplog: pytest.LogCaptureFixture
):
    """All 7 canonical levels are whitelisted, case-normalized, and emitted top-level."""
    with caplog.at_level(logging.WARNING, logger="giskard.llm.translators.openai_chat"):
        payload = OpenAIChatTranslator.to_openai(
            _MODEL, [UserMessage(content="Hi.")], reasoning_effort=raw_effort
        )
    assert payload.get("reasoning_effort") == expected
    assert "reasoning_effort" not in " ".join(r.message for r in caplog.records)
    validate_openai_completion_params(payload)


def test_reasoning_effort_none_omitted():
    """Explicit ``reasoning_effort=None`` omits the field from the wire payload."""
    payload = OpenAIChatTranslator.to_openai(
        _MODEL, [UserMessage(content="Hi.")], reasoning_effort=None
    )
    assert "reasoning_effort" not in payload
    validate_openai_completion_params(payload)


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("openai/gpt-5", id="openai"),
        pytest.param("azure/gpt-5", id="azure"),
        pytest.param("azure_ai/gpt-5", id="azure_ai"),
    ],
)
async def test_client_acompletion_forwards_and_rejects_reasoning_effort(
    model_id: str, monkeypatch: pytest.MonkeyPatch
):
    """``LLMClient.acompletion`` forwards normalized effort and rejects invalid before SDK."""
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

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
    sdk_resp = ChatCompletion(
        id="chatcmpl-1",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Ok"),
            )
        ],
        created=0,
        model="gpt-5",
        object="chat.completion",
    )
    create_mock = AsyncMock(return_value=sdk_resp)
    cast(Any, provider)._client = MagicMock(
        chat=MagicMock(completions=MagicMock(create=create_mock))
    )

    await client.acompletion(
        model_id, [{"role": "user", "content": "Hi"}], reasoning_effort="HIGH"
    )
    assert create_mock.call_args.kwargs["reasoning_effort"] == "high"

    create_mock.reset_mock()
    with pytest.raises(BadRequestError) as exc_info:
        await client.acompletion(
            model_id, [{"role": "user", "content": "Hi"}], reasoning_effort="ultra"
        )
    assert exc_info.value.status_code == 400
    create_mock.assert_not_called()
