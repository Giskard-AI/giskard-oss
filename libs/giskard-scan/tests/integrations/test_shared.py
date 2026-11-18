"""Helpers shared across third-party scanner adapters."""

from giskard.llm.types import (
    AssistantMessage,
    ReasoningContent,
    ReasoningText,
    TextContent,
    UserMessage,
)
from giskard.scan.integrations._shared import trace_from_role_content_turns


def _thought(text: str) -> ReasoningContent:
    return ReasoningContent(
        content=[ReasoningText(text=text)], signature="sig", provider="anthropic"
    )


async def test_trace_from_turns_drops_reasoning_parts_from_assistant_output():
    trace = await trace_from_role_content_turns(
        [
            UserMessage(content="hello"),
            AssistantMessage(
                content=[_thought("Greet back."), TextContent(text="hi there")]
            ),
        ]
    )

    [interaction] = trace.interactions
    assert interaction.inputs == "hello"
    assert interaction.outputs == [TextContent(text="hi there")]


async def test_trace_from_turns_treats_reasoning_only_reply_as_empty():
    trace = await trace_from_role_content_turns(
        [
            UserMessage(content="hello"),
            AssistantMessage(content=[_thought("Nothing to say.")]),
        ]
    )

    [interaction] = trace.interactions
    assert interaction.outputs is None


async def test_trace_from_turns_keeps_string_content_unchanged():
    trace = await trace_from_role_content_turns(
        [UserMessage(content="hello"), AssistantMessage(content="hi there")]
    )

    [interaction] = trace.interactions
    assert interaction.outputs == "hi there"
