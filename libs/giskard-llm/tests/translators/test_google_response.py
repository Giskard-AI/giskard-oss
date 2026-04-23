"""Google Gemini Interactions (response API) translation tests.

Request shape mirrors :meth:`giskard.llm.translators.google_response.GoogleResponseTranslator.to_google`.
"""

from typing import Literal

import pytest
from giskard.llm.translators.google_response import GoogleResponseTranslator
from giskard.llm.types import ResponseEasyInputMessage, ResponseInputItem

from .sdk_payload_validation import validate_google_interaction_params

_MODEL = "gemini-2.0-flash"


def _message(
    role: Literal["user", "assistant", "system", "developer"],
    content: str,
) -> ResponseInputItem:
    """Easy message items with an explicit ``type`` (so system text is not mixed with user)."""
    m: ResponseEasyInputMessage = {"type": "message", "role": role, "content": content}
    return m


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


@pytest.mark.parametrize(
    "instruction_role",
    ["system", "developer"],
)
def test_message_instruction_then_user(
    instruction_role: Literal["system", "developer"],
):
    """System or developer is folded to ``system_instruction``; only user in ``input`` (like chat)."""
    first = (
        _message("system", "You are helpful.")
        if instruction_role == "system"
        else _message("developer", "You are helpful.")
    )
    items: list[ResponseInputItem] = [
        first,
        _message("user", "Hello."),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [
        {**_TEXT, "text": "Hello."},
    ]
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

    assert payload["input"] == [{**_TEXT, "text": "Hello."}]
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

    assert payload["input"] == [{**_TEXT, "text": "Hello."}]
    assert (
        payload.get("system_instruction")
        == "First system instruction.\nSecond system instruction."
    )
    validate_google_interaction_params(payload)


def test_message_user_assistant_user():
    """User and assistant turns map to a flat list of text parts in ``input`` (like ``contents``)."""
    items: list[ResponseInputItem] = [
        _message("user", "First user."),
        _message("assistant", "Assistant reply."),
        _message("user", "Second user."),
    ]
    payload = GoogleResponseTranslator.to_google(_MODEL, items)

    assert payload["input"] == [
        {**_TEXT, "text": "First user."},
        {**_TEXT, "text": "Assistant reply."},
        {**_TEXT, "text": "Second user."},
    ]
    assert "system_instruction" not in payload
    validate_google_interaction_params(payload)
