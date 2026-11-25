"""Tests for WorkflowStep.step_type discriminator (GAP-003)."""

from unittest.mock import AsyncMock, MagicMock

from giskard.agents.generators import BaseGenerator
from giskard.agents.tools import tool
from giskard.agents.workflow import ChatWorkflow, StepType
from giskard.llm.types import (
    AssistantMessage,
    Choice,
    CompletionResponse,
    ToolCall,
    ToolCallFunction,
)


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
            CompletionResponse(
                choices=[
                    Choice(
                        message=AssistantMessage(
                            tool_calls=[
                                ToolCall(
                                    id="tc_1",
                                    function=ToolCallFunction(
                                        name="echo", arguments={"text": "hi"}
                                    ),
                                )
                            ]
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ]
            ),
            CompletionResponse(
                choices=[
                    Choice(
                        message=AssistantMessage(content="Done."),
                        finish_reason="stop",
                        index=0,
                    )
                ]
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
        return_value=CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Hello!"),
                    finish_reason="stop",
                    index=0,
                )
            ]
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
        return_value=CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="World!"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )
    )

    chat = await ChatWorkflow(generator=gen).chat("Hello").run()

    assert not chat.failed
    assert chat.last.content == "World!"


async def test_run_with_no_messages_does_not_raise_indexerror():
    """A ChatWorkflow with no messages added must not crash with IndexError.

    ``_run_tools``'s guard checks ``not chat.last`` before checking whether
    there is anything to process, but ``Chat.last`` indexes ``messages[-1]``,
    which raises ``IndexError`` on an empty list instead of behaving falsy.
    """
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Hello!"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )
    )

    chat = await ChatWorkflow(generator=gen).run()

    assert not chat.failed
    assert chat.last.content == "Hello!"


def _looping_tool_call_response() -> CompletionResponse:
    """A completion that always calls the ``echo`` tool (never finishes)."""
    return CompletionResponse(
        choices=[
            Choice(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            function=ToolCallFunction(
                                name="echo", arguments={"text": "hi"}
                            ),
                        )
                    ]
                ),
                finish_reason="stop",
                index=0,
            )
        ]
    )


def _looping_generator() -> MagicMock:
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(side_effect=lambda *a, **k: _looping_tool_call_response())
    return gen


async def test_default_max_steps_stops_tool_loop():
    """A tool-calling model that never finishes stops after 25 steps by default."""
    gen = _looping_generator()

    collected = []
    async with (
        ChatWorkflow(generator=gen).chat("Echo hi").with_tools(echo).steps() as step_gen
    ):
        async for step in step_gen:
            collected.append(step)

    # 25 steps: 13 completions + 12 tool results, stopping mid-loop.
    assert len(collected) == 25
    assert gen.complete.call_count == 13
    assert collected[-1].step_type == StepType.COMPLETION


async def test_max_steps_none_is_unbounded_with_tools():
    """``max_steps=None`` keeps the old unlimited behaviour for tool workflows."""
    gen = _looping_generator()

    collected = []
    async with (
        ChatWorkflow(generator=gen)
        .chat("Echo hi")
        .with_tools(echo)
        .steps(max_steps=None) as step_gen
    ):
        async for step in step_gen:
            collected.append(step)
            if len(collected) >= 30:
                break

    assert len(collected) == 30
    assert gen.complete.call_count == 15


async def test_default_max_steps_does_not_affect_tool_less_workflow():
    """A workflow without tools still returns after its first completion."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Hello!"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )
    )

    collected = []
    async with ChatWorkflow(generator=gen).chat("Hi").steps() as step_gen:
        async for step in step_gen:
            collected.append(step)

    assert len(collected) == 1
    assert collected[0].step_type == StepType.COMPLETION
    assert gen.complete.call_count == 1


async def test_explicit_max_steps_still_honored():
    """An explicit ``max_steps`` still caps a looping tool workflow."""
    gen = _looping_generator()

    collected = []
    async with (
        ChatWorkflow(generator=gen)
        .chat("Echo hi")
        .with_tools(echo)
        .steps(max_steps=2) as step_gen
    ):
        async for step in step_gen:
            collected.append(step)

    assert len(collected) == 2
    assert gen.complete.call_count == 1
