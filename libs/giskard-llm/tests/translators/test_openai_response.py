"""OpenAI Responses API translation tests.

Request shape mirrors :meth:`giskard.llm.translators.openai_response.OpenAIResponseTranslator.to_openai`.
"""

from typing import Literal

import pytest
from giskard.llm.translators.openai_response import OpenAIResponseTranslator
from giskard.llm.types import ResponseEasyInputMessage, ResponseInputItem

from .sdk_payload_validation import validate_openai_response_params

_MODEL = "gpt-4o-mini"


def _message(
    role: Literal["user", "assistant", "system", "developer"],
    content: str,
) -> ResponseInputItem:
    """Easy message items with an explicit ``type`` (mirrors API easy-input messages)."""
    m: ResponseEasyInputMessage = {"type": "message", "role": role, "content": content}
    return m


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
    first = (
        _message("system", "You are helpful.")
        if instruction_role == "system"
        else _message("developer", "You are helpful.")
    )
    items: list[ResponseInputItem] = [
        first,
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
