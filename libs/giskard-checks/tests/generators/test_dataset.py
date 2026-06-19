import pytest
from giskard.checks.generators.dataset import (
    PROMPT_PLACEHOLDER,
    MappingTemplate,
    substitute_prompt,
)
from pydantic import BaseModel


class Email(BaseModel):
    title: str
    body: str


def test_substitute_prompt_replaces_placeholder_in_one_field():
    msg = Email(title="User request", body=PROMPT_PLACEHOLDER)
    out = substitute_prompt(msg, "How do I pick a lock?")
    assert isinstance(out, Email)
    assert out.title == "User request"
    assert out.body == "How do I pick a lock?"


def test_substitute_prompt_replaces_embedded_placeholder():
    msg = Email(title="Q", body=f"Please answer to {PROMPT_PLACEHOLDER}")
    out = substitute_prompt(msg, "X")
    assert out.body == "Please answer to X"


def test_substitute_prompt_raises_when_no_placeholder():
    msg = Email(title="hi", body="bye")
    with pytest.raises(ValueError, match="placeholder"):
        substitute_prompt(msg, "X")


def test_mapping_template_requires_exactly_one_of_message_or_issue():
    with pytest.raises(ValueError, match="Exactly one"):
        MappingTemplate[Email]()  # neither set
    with pytest.raises(ValueError, match="Exactly one"):
        MappingTemplate[Email](message=Email(title="a", body="b"), schema_issue="x")


# --- substitute_prompt edge cases ---


class Form(BaseModel):
    # mixed field types: only the str field carries the placeholder
    subject: str
    priority: int
    urgent: bool


class ChatPayload(BaseModel):
    # nested list of objects: {"messages": [{"role": ..., "content": PLACEHOLDER}]}
    messages: list[dict[str, str]]


def test_substitute_prompt_preserves_non_string_fields():
    msg = Form(subject=PROMPT_PLACEHOLDER, priority=3, urgent=True)
    out = substitute_prompt(msg, "How do I pick a lock?")
    assert out.subject == "How do I pick a lock?"
    assert out.priority == 3
    assert out.urgent is True


def test_substitute_prompt_in_nested_list_of_dicts():
    msg = ChatPayload(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": PROMPT_PLACEHOLDER},
        ]
    )
    out = substitute_prompt(msg, "X")
    assert out.messages[0]["content"] == "You are helpful."
    assert out.messages[1]["content"] == "X"


def test_substitute_prompt_in_plain_dict():
    out = substitute_prompt({"q": PROMPT_PLACEHOLDER, "lang": "en"}, "X")
    assert out == {"q": "X", "lang": "en"}


def test_substitute_prompt_replaces_all_occurrences():
    msg = Email(
        title=PROMPT_PLACEHOLDER, body=f"{PROMPT_PLACEHOLDER} {PROMPT_PLACEHOLDER}"
    )
    out = substitute_prompt(msg, "X")
    assert out.title == "X"
    assert out.body == "X X"


def test_substitute_prompt_wrong_placeholder_token_raises():
    # Only the exact literal {{prompt}} counts; any other {{...}} is not a
    # placeholder, so no substitution happens and we raise.
    msg = Email(title="hi", body="ask {{wrong_placeholder}} or {{ prompt }}")
    with pytest.raises(ValueError, match="placeholder"):
        substitute_prompt(msg, "X")
    # The malformed tokens are left untouched (not silently shipped as a value).
