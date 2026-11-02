"""Map Anthropic **Messages** API return values to :class:`CompletionResponse`.

This is the ``acompletion`` path. For the OpenAI **Responses** API, see
``test_openai_response.py``; for Google **Interactions** API, see ``test_google_response.py``.

Assistant ``content`` blocks: https://docs.anthropic.com/en/api/messages
"""

from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("anthropic")

from anthropic.types import Message
from anthropic.types.content_block import ContentBlock
from giskard.llm.translators.anthropic import AnthropicChatTranslator
from giskard.llm.types import (
    ChatMessage,
    ReasoningEncryptedDetail,
    ReasoningTextDetail,
    TextContent,
    ToolMessage,
    UserMessage,
)

from .sdk_payload_validation import validate_anthropic_message_create
from .tool_turn_fixtures import TOOL_CALL_ID, TOOL_RESULT_CONTENT, WEATHER_TOOL

pytestmark = pytest.mark.anthropic

_MODEL = "claude-sonnet-4-20250514"

_BASE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _message(data: dict[str, object]) -> Message:
    return Message.model_validate({**_BASE, **data})


def test_from_anthropic_assistant_text():
    """Single `text` block becomes `message.content` and `role` is `assistant`."""
    raw = _message(
        {
            "content": [{"type": "text", "text": "Hello from Claude."}],
        }
    )
    out = AnthropicChatTranslator.from_anthropic(raw)
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5
    assert out.usage.total_tokens == 15
    ch = out.choices[0]
    assert ch.index == 0
    assert ch.finish_reason == "stop"
    assert ch.message.role == "assistant"
    assert ch.message.content == [TextContent(text="Hello from Claude.")]
    assert ch.message.refusal is None
    assert ch.message.tool_calls is None


def test_from_anthropic_refusal_stop():
    """``stop_reason`` ``refusal`` and ``stop_details.explanation`` map to ``message.refusal``."""
    raw = _message(
        {
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {
                "type": "refusal",
                "explanation": "Policy decline.",
            },
        }
    )
    out = AnthropicChatTranslator.from_anthropic(raw)
    ch = out.choices[0]
    assert ch.finish_reason == "stop"
    assert ch.message.content is None
    assert ch.message.refusal == "Policy decline."


def test_from_anthropic_multiple_text_blocks_joined():
    """Several `text` blocks are joined with newlines in `message.content`."""
    raw = _message(
        {
            "content": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
        }
    )
    out = AnthropicChatTranslator.from_anthropic(raw)
    assert out.choices[0].message.content == [
        TextContent(text="Line one."),
        TextContent(text="Line two."),
    ]


def test_from_anthropic_text_and_tool_use():
    """Preamble text plus `tool_use` maps to `content` and `tool_calls` with `finish_reason` `tool_calls`."""
    raw = _message(
        {
            "content": [
                {"type": "text", "text": "I will use a tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
        }
    )
    out = AnthropicChatTranslator.from_anthropic(raw)
    ch = out.choices[0]
    assert ch.finish_reason == "tool_calls"
    msg = ch.message
    assert msg.content == [TextContent(text="I will use a tool.")]
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "toolu_01"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == {"city": "Paris"}


def test_from_anthropic_tool_use_only():
    """Assistant may emit only `tool_use` blocks (no text)."""
    raw = _message(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "add",
                    "input": {"a": 1, "b": 2},
                },
            ],
            "stop_reason": "tool_use",
        }
    )
    out = AnthropicChatTranslator.from_anthropic(raw)
    msg = out.choices[0].message
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "add"
    assert msg.tool_calls[0].function.arguments == {"a": 1, "b": 2}


def test_from_anthropic_refusal_without_details():
    """Refusal with no stop_details still populates message.refusal (#2615)."""
    raw = _message({"content": [], "stop_reason": "refusal", "stop_details": None})
    out = AnthropicChatTranslator.from_anthropic(raw)
    ch = out.choices[0]
    assert ch.finish_reason == "stop"
    assert ch.message.refusal == "refusal"
    assert ch.message.is_refusal


def test_from_anthropic_refusal_category_only():
    """Category without explanation is used as the refusal text (#2615)."""
    raw = _message(
        {
            "content": [{"type": "text", "text": "I can't help with that."}],
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "bio"},
        }
    )
    msg = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    assert msg.refusal == "bio"
    assert msg.is_refusal


def test_from_anthropic_thinking_then_text():
    """Thinking blocks map to ``message.reasoning`` and stay out of ``content`` / ``text``."""
    raw = _message(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "The capital of France is Paris.",
                    "signature": "sig_thinking_block",
                },
                {"type": "text", "text": "Paris is in France."},
            ],
        }
    )
    msg = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    assert msg.reasoning == "The capital of France is Paris."
    assert msg.reasoning_details == [
        ReasoningTextDetail(
            text="The capital of France is Paris.",
            signature="sig_thinking_block",
            format="anthropic-claude-v1",
        )
    ]
    assert msg.content == [TextContent(text="Paris is in France.")]
    assert msg.text == "Paris is in France."


def test_from_anthropic_redacted_thinking_then_text():
    """``redacted_thinking`` is kept opaque in ``reasoning_details``, not ``reasoning``."""
    raw = _message(
        {
            "content": [
                {"type": "redacted_thinking", "data": "opaque-redacted-payload"},
                {"type": "text", "text": "Paris is in France."},
            ],
        }
    )
    msg = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    assert msg.reasoning is None
    assert msg.reasoning_details == [
        ReasoningEncryptedDetail(
            data="opaque-redacted-payload", format="anthropic-claude-v1"
        )
    ]
    assert msg.content == [TextContent(text="Paris is in France.")]
    assert msg.text == "Paris is in France."


def test_from_anthropic_thinking_and_tool_use():
    """Thinking ahead of a tool call maps to ``reasoning`` and still yields the tool call."""
    raw = _message(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should check the weather.",
                    "signature": "sig_xyz",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_03",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
        }
    )
    choice = AnthropicChatTranslator.from_anthropic(raw).choices[0]
    msg = choice.message
    assert choice.finish_reason == "tool_calls"
    assert msg.reasoning == "I should check the weather."
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == {"city": "Paris"}


def test_from_anthropic_redacted_thinking_only_is_dropped_not_raised():
    """A response with only ``redacted_thinking`` does not crash and has no visible output."""
    raw = _message(
        {"content": [{"type": "redacted_thinking", "data": "opaque-encrypted-data"}]}
    )
    msg = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    assert msg.reasoning is None
    assert msg.reasoning_details == [
        ReasoningEncryptedDetail(
            data="opaque-encrypted-data", format="anthropic-claude-v1"
        )
    ]
    assert msg.content is None
    assert msg.tool_calls is None


def test_from_anthropic_text_only_has_no_reasoning():
    """A text-only response leaves ``message.reasoning`` / ``reasoning_details`` unset."""
    raw = _message({"content": [{"type": "text", "text": "Hello from Claude."}]})
    msg = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    assert msg.reasoning is None
    assert msg.reasoning_details is None
    assert msg.text == "Hello from Claude."


def test_block_content_to_giskard_unknown_type_raises():
    """Unknown content-block types still raise so new Anthropic types stay visible."""
    from giskard.llm.errors import UnsupportedContentError

    with pytest.raises(UnsupportedContentError) as exc_info:
        _ = AnthropicChatTranslator.block_content_to_giskard(
            cast(ContentBlock, cast(object, SimpleNamespace(type="compaction")))
        )
    assert exc_info.value.content_type == "compaction"


_SERVER_TOOL_USE = {
    "type": "server_tool_use",
    "id": "srvtoolu_01",
    "name": "web_search",
    "input": {"query": "weather in Paris"},
}


def test_from_anthropic_unsupported_block_raises():
    """Blocks we cannot represent raise :class:`UnsupportedContentError` by default."""
    from giskard.llm.errors import UnsupportedContentError

    raw = _message({"content": [_SERVER_TOOL_USE, {"type": "text", "text": "Answer."}]})
    with pytest.raises(UnsupportedContentError) as exc_info:
        AnthropicChatTranslator.from_anthropic(raw)
    assert exc_info.value.content_type == "server_tool_use"
    assert exc_info.value.provider == "anthropic"


def test_from_anthropic_unsupported_block_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
):
    """With ``ignore_unsupported_content`` the block is dropped and a warning logged."""
    raw = _message({"content": [_SERVER_TOOL_USE, {"type": "text", "text": "Answer."}]})
    with caplog.at_level("WARNING"):
        msg = (
            AnthropicChatTranslator.from_anthropic(raw, ignore_unsupported_content=True)
            .choices[0]
            .message
        )
    assert msg.content == [TextContent(text="Answer.")]
    assert "server_tool_use" in caplog.text


def test_thinking_tool_turn_passed_back_unchanged_replays_verbatim():
    """A thinking + tool_use response, passed back as-is, replays its blocks in order."""
    raw = _message(
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "Need weather.", "signature": "s1"},
                {"type": "redacted_thinking", "data": "opaque"},
                {
                    "type": "tool_use",
                    "id": TOOL_CALL_ID,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        }
    )
    assistant = AnthropicChatTranslator.from_anthropic(raw).choices[0].message
    messages: list[ChatMessage] = [
        UserMessage(content="What's the weather in Paris?"),
        assistant,
        ToolMessage(content=TOOL_RESULT_CONTENT, tool_call_id=TOOL_CALL_ID),
    ]
    payload = AnthropicChatTranslator.to_anthropic(
        _MODEL, messages, tools=[WEATHER_TOOL], thinking={"type": "adaptive"}
    )

    assert payload["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "Need weather.", "signature": "s1"},
        {"type": "redacted_thinking", "data": "opaque"},
        {
            "type": "tool_use",
            "id": TOOL_CALL_ID,
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ]
    validate_anthropic_message_create(payload)
