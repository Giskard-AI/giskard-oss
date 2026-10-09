"""Map OpenAI **Chat Completions** return values to :class:`CompletionResponse`.

This is the ``acompletion`` / :class:`ChatCompletion` path. For the **Responses** API: request
``to_openai`` in ``test_openai_response.py``; return ``Response`` -> ``ResponseResult`` in
``test_openai_response_return.py``.

Assistant ``message`` fields: https://platform.openai.com/docs/api-reference/chat/object
"""

import json

import pytest

pytest.importorskip("openai")

from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

pytestmark = pytest.mark.openai

_MODEL = "gpt-4o-mini"


def test_from_openai_assistant_text_message():
    """Maps ``choices[].message`` with string ``content`` (typical text reply)."""
    raw = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello, world.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5
    assert out.usage.total_tokens == 15
    assert len(out.choices) == 1
    ch = out.choices[0]
    assert ch.index == 0
    assert ch.finish_reason == "stop"
    assert ch.message.role == "assistant"
    assert ch.message.content == "Hello, world."
    assert ch.message.refusal is None
    assert ch.message.tool_calls is None


def test_from_openai_assistant_message_omit_usage():
    """``usage`` may be absent on the completion object."""
    raw = ChatCompletion(
        id="chatcmpl-test2",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="No usage.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert out.usage is None
    assert out.choices[0].message.content == "No usage."


def test_from_openai_assistant_refusal():
    """Maps ``message.refusal`` when the model returns a policy refusal (no ``content``)."""
    raw = ChatCompletion(
        id="chatcmpl-refusal",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    refusal="I'm sorry, I can't assist with that.",
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content is None
    assert msg.refusal == "I'm sorry, I can't assist with that."


def test_from_openai_assistant_text_and_tool_calls():
    """Assistant `message` may include both `content` and `tool_calls` (e.g. a short preamble)."""
    raw = ChatCompletion(
        id="chatcmpl-tools",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="I will use the function.",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments=json.dumps({"city": "Paris"}),
                            ),
                        )
                    ],
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content == "I will use the function."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_abc"
    assert msg.tool_calls[0].type == "function"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == {"city": "Paris"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_from_openai_assistant_tool_calls_only():
    """Tool-only turn: `content` is null, `tool_calls` populated, `finish_reason` is `tool_calls`."""
    raw = ChatCompletion(
        id="chatcmpl-toolonly",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="f",
                                arguments=json.dumps({"x": 1}),
                            ),
                        )
                    ],
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    msg = out.choices[0].message
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.arguments == {"x": 1}


def test_from_openai_two_choices():
    """Maps multiple `choices` when `n` > 1 (each index and message preserved)."""
    raw = ChatCompletion(
        id="chatcmpl-n2",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="A"),
            ),
            Choice(
                index=1,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="B"),
            ),
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )
    out = OpenAIChatTranslator.from_openai(raw)
    assert len(out.choices) == 2
    assert out.choices[0].index == 0
    assert out.choices[0].message.content == "A"
    assert out.choices[1].index == 1
    assert out.choices[1].message.content == "B"


# -- Reasoning (OpenAI-compatible extension: OpenRouter / vLLM / DeepSeek) ----------


def _completion_with_message(**extras: object) -> ChatCompletion:
    """Build a completion whose message carries non-official extras, as the SDK keeps them."""
    return ChatCompletion(
        id="chatcmpl-reasoning",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage.model_validate(
                    {"role": "assistant", "content": "42", **extras}
                ),
            )
        ],
        created=0,
        model=_MODEL,
        object="chat.completion",
    )


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_from_openai_reasoning_plaintext(field: str):
    """vLLM ``reasoning`` and DeepSeek ``reasoning_content`` both land on ``reasoning``."""
    raw = _completion_with_message(**{field: "Let me think step by step."})
    msg = OpenAIChatTranslator.from_openai(raw).choices[0].message
    assert msg.reasoning == "Let me think step by step."
    assert msg.text == "42"
    assert "think step by step" not in msg.transcript


def test_from_openai_reasoning_details():
    """Each OpenRouter ``reasoning_details`` item type parses with its fields preserved."""
    from giskard.llm.types import (
        ReasoningEncryptedDetail,
        ReasoningSummaryDetail,
        ReasoningTextDetail,
    )

    raw = _completion_with_message(
        reasoning="Thinking...",
        reasoning_details=[
            {
                "type": "reasoning.text",
                "text": "Thinking...",
                "signature": "sig",
                "id": "r1",
                "format": "anthropic-claude-v1",
                "index": 0,
            },
            {
                "type": "reasoning.summary",
                "summary": "Short summary",
                "id": None,
                "format": "openai-responses-v1",
            },
            {
                "type": "reasoning.encrypted",
                "data": "opaque",
                "id": "r3",
                "format": "google-gemini-v1",
            },
        ],
    )
    msg = OpenAIChatTranslator.from_openai(raw).choices[0].message
    assert msg.reasoning_details == [
        ReasoningTextDetail(
            text="Thinking...",
            signature="sig",
            id="r1",
            format="anthropic-claude-v1",
            index=0,
        ),
        ReasoningSummaryDetail(summary="Short summary", format="openai-responses-v1"),
        ReasoningEncryptedDetail(data="opaque", id="r3", format="google-gemini-v1"),
    ]
    assert msg.text == "42"
    assert "Thinking" not in msg.transcript
    assert "Short summary" not in msg.transcript


def test_from_openai_unknown_reasoning_detail_raises():
    """An unknown ``reasoning_details`` type is not silently dropped by default."""
    from giskard.llm.errors import UnsupportedContentError

    raw = _completion_with_message(
        reasoning_details=[{"type": "reasoning.future", "payload": "x"}]
    )
    with pytest.raises(UnsupportedContentError, match="reasoning.future") as exc_info:
        OpenAIChatTranslator.from_openai(raw)
    assert exc_info.value.status_code == 0
    assert exc_info.value.content_type == "reasoning_details:reasoning.future"


def test_from_openai_unknown_reasoning_detail_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
):
    """With ``ignore_unsupported_content`` the unknown item is dropped and a warning logged."""
    raw = _completion_with_message(
        reasoning_details=[
            {"type": "reasoning.future", "payload": "x"},
            {"type": "reasoning.encrypted", "data": "opaque", "format": "unknown"},
        ]
    )
    with caplog.at_level("WARNING"):
        msg = (
            OpenAIChatTranslator.from_openai(raw, ignore_unsupported_content=True)
            .choices[0]
            .message
        )
    assert msg.reasoning_details is not None
    assert [d.type for d in msg.reasoning_details] == ["reasoning.encrypted"]
    assert "reasoning.future" in caplog.text
