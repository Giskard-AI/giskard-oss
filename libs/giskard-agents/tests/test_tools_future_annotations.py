"""Tests for tools defined in a module using PEP 563 string annotations."""

from __future__ import annotations

from giskard.agents.context import RunContext
from giskard.agents.tools import tool
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


@tool
def count_tool(context: RunContext, increment: int = 1) -> int:
    """Count the number of times this tool has been called.

    Parameters
    ----------
    context : RunContext
        The run context to store state.
    increment : int, optional
        How much to increment the counter by, by default 1.
    """
    current_count = context.get("call_count", 0)
    new_count = current_count + increment
    context.set("call_count", new_count)
    return new_count


def test_run_context_param_is_detected():
    assert count_tool.run_context_param == "context"


def test_run_context_is_not_exposed_to_the_model():
    schema = count_tool.parameters_schema
    assert "increment" in schema["properties"]
    assert "context" not in schema["properties"]
    assert "context" not in schema.get("required", [])


async def test_run_context_is_injected():
    context = RunContext()

    await count_tool.run({"increment": 2}, ctx=context)

    assert context.get("call_count") == 2


def test_tool_with_a_model_annotation():
    @tool
    def move(point: Point) -> str:
        """Move to a point.

        Parameters
        ----------
        point : Point
            Target point.
        """
        return f"{point.x},{point.y}"

    assert "point" in move.parameters_schema["properties"]
