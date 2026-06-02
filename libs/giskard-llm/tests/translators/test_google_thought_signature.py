"""Tests for Gemini ``thought_signature`` round-trip fidelity.

Gemini 3+ thinking models return a ``thought_signature`` on text ``Part`` objects.
``giskard-llm`` must preserve this field when converting to/from ``TextContent`` so
that subsequent requests can replay the signature back to the model.

Docs: https://ai.google.dev/gemini-api/docs/thought-signatures
"""

from types import SimpleNamespace
from typing import Any

import pytest
from giskard.llm.translators.google_chat import GoogleChatTranslator
from giskard.llm.types import AssistantMessage, TextContent, UserMessage


def _make_part(text: str, thought_signature: str | None = None) -> Any:
    """Build a duck-typed Part-like object without requiring the google.genai SDK."""
    part = SimpleNamespace(text=text, function_call=None)
    if thought_signature is not None:
        part.thought_signature = thought_signature
    return part


def test_part_content_to_giskard_preserves_thought_signature():
    """``part_content_to_giskard`` stores ``thought_signature`` in ``TextContent``."""
    part = _make_part(text="I'm thinking…", thought_signature="sig-abc123")
    result = GoogleChatTranslator.part_content_to_giskard(
        part, num_messages=1, part_index=0
    )

    assert isinstance(result, TextContent)
    assert result.text == "I'm thinking…"
    assert result.thought_signature == "sig-abc123"


def test_part_content_to_giskard_no_thought_signature():
    """Parts without ``thought_signature`` produce ``TextContent`` with ``None``."""
    part = _make_part(text="Normal reply.")
    result = GoogleChatTranslator.part_content_to_giskard(
        part, num_messages=1, part_index=0
    )

    assert isinstance(result, TextContent)
    assert result.text == "Normal reply."
    assert result.thought_signature is None


def test_serialize_text_content_includes_thought_signature():
    """``to_google`` round-trips ``thought_signature`` back into the Google payload."""
    messages = [
        UserMessage(content="Solve this step by step."),
        AssistantMessage(
            content=[TextContent(text="Let me think…", thought_signature="sig-xyz")]
        ),
        UserMessage(content="Continue."),
    ]
    payload = GoogleChatTranslator.to_google("gemini-2.0-flash", messages)

    model_parts = payload["contents"][1]["parts"]  # type: ignore[index]
    assert model_parts == [{"text": "Let me think…", "thought_signature": "sig-xyz"}]


def test_serialize_text_content_omits_null_thought_signature():
    """``to_google`` omits ``thought_signature`` when it is ``None``."""
    messages = [
        UserMessage(content="Hello."),
        AssistantMessage(content=[TextContent(text="Hi there.")]),
    ]
    payload = GoogleChatTranslator.to_google("gemini-2.0-flash", messages)

    model_parts = payload["contents"][1]["parts"]  # type: ignore[index]
    assert model_parts == [{"text": "Hi there."}]
    assert "thought_signature" not in model_parts[0]


# ---------------------------------------------------------------------------
# The tests below require the google.genai SDK (installed in CI with extras).
# They validate the full from_google -> to_google round trip using real SDK types.
# ---------------------------------------------------------------------------

pytest.importorskip("google.genai")  # skip the rest if SDK not installed

from google.genai import types  # noqa: E402  (conditional import after skip)

pytestmark = pytest.mark.google

_MODEL = "gemini-2.0-flash"


def _raw(data: dict[str, object]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(data)


def test_from_google_preserves_thought_signature_when_sdk_supports_it():
    """``from_google`` stores ``thought_signature`` when the SDK Part carries it.

    If the installed SDK version does not expose ``thought_signature`` on Part,
    the field is silently set to ``None`` — no error is raised.
    """
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Thinking output."}],
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
    )
    # Inject thought_signature directly onto the Part object (SDK may not yet
    # expose this field via model_validate; use setattr for forward compatibility).
    part = raw.candidates[0].content.parts[0]  # type: ignore[index]
    object.__setattr__(part, "thought_signature", "sig-round-trip")

    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    content = out.choices[0].message.content
    assert content is not None
    assert isinstance(content[0], TextContent)
    assert content[0].thought_signature == "sig-round-trip"


def test_from_google_round_trips_thought_signature_to_google():
    """A Gemini response with ``thought_signature`` round-trips back to Google payload."""
    raw = _raw(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Agentic reasoning step."}],
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
    )
    part = raw.candidates[0].content.parts[0]  # type: ignore[index]
    object.__setattr__(part, "thought_signature", "sig-fidelity")

    out = GoogleChatTranslator.from_google(raw, _MODEL, 1)
    assistant_msg = out.choices[0].message

    # Now replay the assistant message back to Google
    follow_up = [
        UserMessage(content="Next step."),
        assistant_msg,
        UserMessage(content="Continue."),
    ]
    payload = GoogleChatTranslator.to_google(_MODEL, follow_up)

    model_parts = payload["contents"][1]["parts"]  # type: ignore[index]
    assert model_parts == [
        {"text": "Agentic reasoning step.", "thought_signature": "sig-fidelity"}
    ]
