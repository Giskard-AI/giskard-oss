"""Tests for WorkflowStep.step_type discriminator (GAP-003)."""

from unittest.mock import AsyncMock, MagicMock

from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator
from giskard.agents.generators.base import Response
from giskard.agents.tools import Function, ToolCall, tool
from giskard.agents.workflow import ChatWorkflow, StepType


@tool
def echo(text: str) -> str:
    """Echo text back.

    Parameters
    ----------
    text : str
        Text to echo.
    """
    return text


async def test_steps_have_correct_step_type():
    """Tool result steps are TOOL_RESULT, final completion is COMPLETION."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        side_effect=[
            Response(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            function=Function(name="echo", arguments='{"text": "hi"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            ),
            Response(
                message=Message(role="assistant", content="Done."),
                finish_reason="stop",
            ),
        ]
    )

    collected = []
    async with (
        ChatWorkflow(generator=gen)
        .chat("Echo hi")
        .with_tools(echo)
        .steps(max_steps=10) as step_gen
    ):
        async for step in step_gen:
            collected.append(step)

    assert len(collected) == 3
    assert collected[0].step_type == StepType.COMPLETION
    assert collected[1].step_type == StepType.TOOL_RESULT
    assert collected[2].step_type == StepType.COMPLETION


async def test_step_type_completion_when_no_tools():
    """A simple completion without tools yields a single COMPLETION step."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=Response(
            message=Message(role="assistant", content="Hello!"),
            finish_reason="stop",
        )
    )

    collected = []
    async with ChatWorkflow(generator=gen).chat("Hi").steps(max_steps=10) as step_gen:
        async for step in step_gen:
            collected.append(step)

    assert len(collected) == 1
    assert collected[0].step_type == StepType.COMPLETION


async def test_run_returns_successful_chat():
    """run() returns a Chat with the assistant's response."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=Response(
            message=Message(role="assistant", content="World!"),
            finish_reason="stop",
        )
    )

    chat = await ChatWorkflow(generator=gen).chat("Hello").run()

    assert not chat.failed
    assert chat.last.content == "World!"
