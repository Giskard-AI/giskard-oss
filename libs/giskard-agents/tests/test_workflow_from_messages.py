"""Tests for ChatWorkflow.from_messages() (GAP-002)."""

from unittest.mock import AsyncMock, MagicMock

from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator
from giskard.agents.generators.base import Response
from giskard.agents.workflow import ChatWorkflow


def _simple_generator() -> MagicMock:
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=Response(
            message=Message(role="assistant", content="OK"),
            finish_reason="stop",
        )
    )
    return gen


async def test_from_messages_with_dicts():
    """from_messages hydrates dicts into Message objects."""
    gen = _simple_generator()
    wf = ChatWorkflow.from_messages(
        gen,
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    )

    assert len(wf.messages) == 2
    assert all(isinstance(m, Message) for m in wf.messages)


async def test_from_messages_preserves_roles_and_content():
    """Roles and content survive the round-trip."""
    gen = _simple_generator()
    wf = ChatWorkflow.from_messages(
        gen,
        [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User says hello."},
            {"role": "assistant", "content": "Assistant responds."},
        ],
    )

    assert wf.messages[0].role == "system"
    assert wf.messages[0].content == "System prompt."
    assert wf.messages[1].role == "user"
    assert wf.messages[1].content == "User says hello."
    assert wf.messages[2].role == "assistant"
    assert wf.messages[2].content == "Assistant responds."


async def test_from_messages_with_tool_calls():
    """Dicts containing tool_calls are correctly parsed."""
    gen = _simple_generator()
    wf = ChatWorkflow.from_messages(
        gen,
        [
            {"role": "user", "content": "Do something."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ],
    )

    assert len(wf.messages) == 3
    assert wf.messages[1].tool_calls is not None
    assert len(wf.messages[1].tool_calls) == 1
    assert wf.messages[1].tool_calls[0].function.name == "my_tool"
    assert wf.messages[2].role == "tool"
    assert wf.messages[2].tool_call_id == "tc_1"


async def test_from_messages_runs_workflow():
    """A workflow created with from_messages can be run normally."""
    gen = _simple_generator()
    chat = await ChatWorkflow.from_messages(
        gen,
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
    ).run()

    assert not chat.failed
    assert chat.last.content == "OK"
