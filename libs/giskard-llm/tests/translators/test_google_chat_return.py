"""Map Gemini **generateContent** return values to :class:`CompletionResponse`.

This is the ``acompletion`` path. For the **Interactions** API: request ``to_google`` in
``test_google_response.py``; return ``Interaction`` -> ``ResponseResult`` in
``test_google_response_return.py``.

``candidates`` / ``content.parts``: https://ai.google.dev/api/generate-content#method:-models.generatecontent
"""

import pytest

pytest.importorskip("google.genai")

from giskard.llm.translators.google_chat import GoogleChatTranslator
from giskard.llm.types import TextContent
from google.genai import types

pytestmark = pytest.mark.google

_MODEL = "gemini-3.5-flash"


def _raw(data: dict[str, object]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(data)


def test_from_google_assistant_text():
    """`parts` with a single `text` field map to `message.content`."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello from Gemini."}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 3,
                "candidates_token_count": 4,
                "total_token_count": 7,
            },
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    assert out.model == _MODEL
    assert out.usage is not None
    assert out.usage.input_tokens == 3
    assert out.usage.output_tokens == 4
    assert out.usage.total_tokens == 7
    ch = out.choices[0]
    assert ch.index == 0
    assert ch.message.role == "assistant"
    assert ch.message.content == [TextContent(text="Hello from Gemini.")]
    assert ch.message.tool_calls is None
    assert ch.finish_reason == "stop"


def test_from_google_multiple_text_parts_joined():
    """Multiple `text` parts are joined with newlines, matching multi-part model output."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "First."},
                            {"text": "Second."},
                        ]
                    }
                }
            ],
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    assert out.choices[0].message.content == [
        TextContent(text="First."),
        TextContent(text="Second."),
    ]


def test_from_google_text_and_function_call():
    """Model turn may mix `text` and `function_call` parts (preamble + tool)."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I will look up the weather."},
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "Paris"},
                                }
                            },
                        ]
                    }
                }
            ],
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    ch = out.choices[0]
    assert ch.finish_reason == "tool_calls"
    msg = ch.message
    assert msg.content == [TextContent(text="I will look up the weather.")]
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].type == "function"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == {"city": "Paris"}


def test_from_google_function_call_captures_thought_signature():
    """A ``function_call`` part's ``thought_signature`` is captured on the tool call."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "Paris"},
                                },
                                "thought_signature": b"real-signature-bytes",
                            },
                        ]
                    }
                }
            ],
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    tool_calls = out.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].thought_signature == b"real-signature-bytes"


def test_from_google_function_call_without_signature_is_none():
    """A ``function_call`` part with no signature leaves ``thought_signature`` unset."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "Paris"},
                                }
                            },
                        ]
                    }
                }
            ],
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    tool_calls = out.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].thought_signature is None


def test_from_google_text_part_captures_thought_signature():
    """A text part's ``thought_signature`` is captured on the text content."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Let me think step by step.",
                                "thought_signature": b"text-signature-bytes",
                            },
                        ]
                    }
                }
            ],
        }
    )
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    assert out.choices[0].message.content == [
        TextContent(
            text="Let me think step by step.",
            thought_signature=b"text-signature-bytes",
        )
    ]


def test_from_google_empty_candidates():
    """No `candidates` yields an empty `choices` list; only `model` is set."""
    raw = _raw({"candidates": []})
    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    assert out.choices == []
    assert out.model == _MODEL
