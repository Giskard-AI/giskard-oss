"""Tests for RunContext functionality."""

from unittest.mock import MagicMock

from giskard.agents.chat import Message
from giskard.agents.context import RunContext
from giskard.agents.generators import BaseGenerator
from giskard.agents.generators.base import Response
from giskard.agents.tools import Function, ToolCall, tool
from giskard.agents.workflow import ChatWorkflow


@tool
def count_tool(context: RunContext, increment: int = 1) -> int:
    """Count the number of times this tool has been called.

    Parameters
    ----------
    increment : int, optional
        How much to increment the counter by, by default 1.
    context : RunContext
        The run context to store state.
    """
    current_count = context.get("call_count", 0)
    new_count = current_count + increment
    context.set("call_count", new_count)
    return new_count


def test_run_context_basic_functionality():
    """Test basic RunContext operations."""
    context = RunContext()

    # Test set and get
    context.set("test_key", "test_value")
    assert context.get("test_key") == "test_value"

    # Test get with default
    assert context.get("nonexistent", "default") == "default"
    assert context.get("nonexistent") is None

    # Test has
    assert context.has("test_key") is True
    assert context.has("nonexistent") is False

    # Test clear
    context.clear()
    assert context.has("test_key") is False


async def test_tool_context_injection():
    context = RunContext()

    assert count_tool.run_context_param == "context"

    # One
    await count_tool.run({"increment": 1}, ctx=context)
    assert context.get("call_count") == 1

    # Two
    await count_tool.run({"increment": 2}, ctx=context)
    assert context.get("call_count") == 3


def test_tool_context_not_in_params():
    assert count_tool.run_context_param == "context"
    assert "increment" in count_tool.parameters_schema["properties"]
    assert "context" not in count_tool.parameters_schema["properties"]
    assert "ctx" not in count_tool.parameters_schema["properties"]


async def test_pipeline_calls_context():
    generator = MagicMock(spec=BaseGenerator)
    generator.complete.return_value = Response(
        message=Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="1",
                    function=Function(name="count_tool", arguments='{"increment": 1}'),
                )
            ],
        ),
        finish_reason="tool_calls",
    )

    pipeline = (
        ChatWorkflow(generator=generator)
        .with_inputs(name="TestBot")
        .with_tools(count_tool)
        .chat("Increment the count by 1")
    )

    # First step will generate the tool_call request, second step will call the tool and return the result.
    chat = await pipeline.run(max_steps=2)

    assert chat.context.get("call_count") == 1
    assert chat.context.inputs["name"] == "TestBot"
