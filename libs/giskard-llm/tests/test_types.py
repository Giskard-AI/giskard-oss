import pytest
from giskard.llm.types import (
    AssistantMessage,
    Choice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    FunctionMessage,
    ToolCall,
    ToolCallFunction,
)


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


def test_assistant_reasoning_never_in_text_or_transcript():
    """Checks and judges read ``text`` / ``transcript``: reasoning must never leak there."""
    from giskard.llm.types import ReasoningSummaryDetail, ReasoningTextDetail

    msg = AssistantMessage(
        content="The answer is 42.",
        reasoning="SECRET chain of thought",
        reasoning_details=[
            ReasoningTextDetail(text="SECRET detail text"),
            ReasoningSummaryDetail(summary="SECRET summary"),
        ],
    )
    assert msg.text == "The answer is 42."
    assert "SECRET" not in msg.transcript

    reasoning_only = AssistantMessage(reasoning="SECRET chain of thought")
    assert reasoning_only.text is None
    assert "SECRET" not in reasoning_only.transcript


def test_assistant_reasoning_content_alias_and_discriminated_details():
    """``reasoning_content`` is accepted as an alias; details dispatch on ``type``."""
    from giskard.llm.types import ReasoningEncryptedDetail

    msg = AssistantMessage.model_validate(
        {
            "role": "assistant",
            "reasoning_content": "thinking",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "x"}],
        }
    )
    assert msg.reasoning == "thinking"
    assert msg.reasoning_details == [ReasoningEncryptedDetail(data="x")]
    assert msg.model_dump()["reasoning"] == "thinking"


def test_assistant_null_reasoning_does_not_shadow_reasoning_content():
    """A ``reasoning: null`` key must not hide a populated ``reasoning_content``."""
    msg = AssistantMessage.model_validate(
        {"role": "assistant", "reasoning": None, "reasoning_content": "thinking"}
    )
    assert msg.reasoning == "thinking"


def test_response_result_output_text_excludes_reasoning():
    from giskard.llm.types import ResponseResult

    result = ResponseResult.model_validate(
        {
            "id": "resp_1",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "SECRET"}],
                    "content": [{"type": "reasoning_text", "text": "SECRET"}],
                },
                {"type": "message", "content": "42"},
            ],
        }
    )
    assert result.output_text == "42"
    assert [r.id for r in result.reasoning] == ["rs_1"]


def test_reasoning_effort_exports_and_levels():
    """ReasoningEffort is exported from both giskard.llm and giskard.llm.types."""
    from typing import get_args

    from giskard.llm import ReasoningEffort as TopLevelReasoningEffort
    from giskard.llm.types import ReasoningEffort

    assert TopLevelReasoningEffort is ReasoningEffort
    assert get_args(ReasoningEffort) == (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        pytest.param("none", "none", id="none-lower"),
        pytest.param("minimal", "minimal", id="minimal-lower"),
        pytest.param("low", "low", id="low-lower"),
        pytest.param("medium", "medium", id="medium-lower"),
        pytest.param("high", "high", id="high-lower"),
        pytest.param("xhigh", "xhigh", id="xhigh-lower"),
        pytest.param("max", "max", id="max-lower"),
        pytest.param("HIGH", "high", id="upper-case"),
        pytest.param("Medium", "medium", id="title-case"),
        pytest.param("xHiGh", "xhigh", id="mixed-case"),
    ],
)
def test_normalized_reasoning_effort_accepts_and_lowercases(
    raw_value: str, expected: str
):
    from giskard.llm.types import NormalizedReasoningEffort, normalize_reasoning_effort
    from pydantic import BaseModel

    class _Cfg(BaseModel):
        reasoning_effort: NormalizedReasoningEffort | None = None

    assert (
        _Cfg.model_validate({"reasoning_effort": raw_value}).reasoning_effort
        == expected
    )
    assert normalize_reasoning_effort(raw_value) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param("ultra", id="unknown-level"),
        pytest.param("", id="empty-string"),
        pytest.param(" high ", id="whitespace-padded"),
        pytest.param(123, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(["high"], id="list"),
        pytest.param(None, id="none-non-optional"),
    ],
)
def test_normalized_reasoning_effort_rejects_invalid(invalid_value: object):
    from giskard.llm.types import NormalizedReasoningEffort, normalize_reasoning_effort
    from pydantic import BaseModel, ValidationError

    class _Cfg(BaseModel):
        reasoning_effort: NormalizedReasoningEffort

    with pytest.raises(ValidationError):
        _Cfg.model_validate({"reasoning_effort": invalid_value})

    with pytest.raises(ValueError):
        normalize_reasoning_effort(invalid_value)


def test_normalized_reasoning_effort_optional_accepts_none():
    from giskard.llm.types import NormalizedReasoningEffort
    from pydantic import BaseModel

    class _Cfg(BaseModel):
        reasoning_effort: NormalizedReasoningEffort | None = None

    assert _Cfg().reasoning_effort is None
    assert _Cfg.model_validate({"reasoning_effort": None}).reasoning_effort is None


@pytest.mark.parametrize(
    ("kwargs", "expected_value", "expected_in_dump"),
    [
        pytest.param({}, None, False, id="default-none-omitted"),
        pytest.param(
            {"reasoning_tokens": None}, None, False, id="explicit-none-omitted"
        ),
        pytest.param({"reasoning_tokens": 0}, 0, True, id="zero-preserved"),
        pytest.param({"reasoning_tokens": 42}, 42, True, id="positive-preserved"),
    ],
)
def test_usage_reasoning_tokens_default_and_serialization(
    kwargs: dict[str, object], expected_value: int | None, expected_in_dump: bool
):
    """``Usage.reasoning_tokens`` defaults to ``None`` and is omitted when ``None``."""
    from giskard.llm.types import Usage

    usage = Usage.model_validate(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, **kwargs}
    )
    assert usage.reasoning_tokens == expected_value
    dump = usage.model_dump()
    assert ("reasoning_tokens" in dump) is expected_in_dump
    if expected_in_dump:
        assert dump["reasoning_tokens"] == expected_value
