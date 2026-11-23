import json

import pytest
from giskard.llm.types import (
    AssistantMessage,
    Choice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    FunctionMessage,
    TextContent,
    ToolCall,
    ToolCallFunction,
)
from pydantic import ValidationError


def test_completion_response_model_dump():
    resp = CompletionResponse(
        choices=[
            Choice(
                message=AssistantMessage(content="Hello"),
                finish_reason="stop",
            )
        ],
        model="gpt-4o",
    )
    dump = resp.model_dump()
    assert dump["choices"][0]["message"]["role"] == "assistant"
    assert dump["choices"][0]["message"]["content"] == "Hello"
    assert dump["choices"][0]["finish_reason"] == "stop"
    assert dump["model"] == "gpt-4o"


def test_choice_message_excludes_none():
    msg = AssistantMessage(content="Hello")
    dump = msg.model_dump()
    assert "tool_calls" not in dump


def test_assistant_message_transcript_single_role_prefix():
    assert AssistantMessage(content="Hello").transcript == "[assistant]: Hello"


def test_assistant_message_transcript_with_tool_calls_no_duplicated_prefix():
    msg = AssistantMessage(
        content="OK",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="add", arguments={"a": 1}),
            )
        ],
    )
    t = msg.transcript
    assert t.count("[assistant]:") == 1, t
    assert t.startswith("[assistant]: OK\n>")


def test_choice_message_includes_typed_tool_calls():
    msg = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="add", arguments={"a": 1, "b": 2}),
            )
        ],
    )
    dump = msg.model_dump()
    assert dump["tool_calls"] is not None
    assert len(dump["tool_calls"]) == 1
    assert dump["tool_calls"][0]["function"]["name"] == "add"


def test_tool_call_model():
    tc = ToolCall(
        id="call_1",
        function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
    )
    assert tc.id == "call_1"
    assert tc.type == "function"
    assert tc.function.name == "get_weather"


def test_function_message_transcript_none_content():
    assert FunctionMessage(name="fn").transcript == "[function]: empty"


def test_function_message_transcript_with_content():
    assert (
        FunctionMessage(name="fn", content="result").transcript == "[function]: result"
    )


def test_embedding_response():
    resp = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        ]
    )
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]


def test_thought_signature_survives_dict_and_json_round_trip():
    """Binary thought signatures and provider tags round-trip cleanly through dict and JSON."""
    text_sig = b"\x80\xff\x00text-signature"
    tool_sig = b"\x00\xfe\x81tool-signature"
    msg = AssistantMessage(
        content=[
            TextContent(
                text="Checking weather.",
                thought_signature=text_sig,
                thought_signature_provider="google",
            )
        ],
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolCallFunction(
                    name="get_weather", arguments={"city": "Paris"}
                ),
                thought_signature=tool_sig,
                thought_signature_provider="google",
            )
        ],
    )

    dumped = msg.model_dump()
    # model_dump() must already be JSON-safe even for non-UTF-8 signature bytes.
    raw_json = json.dumps(dumped)
    from_dict = AssistantMessage.model_validate(json.loads(raw_json))
    from_json = AssistantMessage.model_validate_json(msg.model_dump_json())

    for restored in (from_dict, from_json):
        assert isinstance(restored.content, list)
        assert isinstance(restored.content[0], TextContent)
        assert restored.content[0].thought_signature == text_sig
        assert restored.content[0].thought_signature_provider == "google"
        assert restored.tool_calls is not None
        assert restored.tool_calls[0].thought_signature == tool_sig
        assert restored.tool_calls[0].thought_signature_provider == "google"


def test_thought_signature_rejects_invalid_base64():
    with pytest.raises(ValidationError):
        TextContent(text="Hi", thought_signature="not-valid-base64!!!")  # type: ignore[arg-type]
