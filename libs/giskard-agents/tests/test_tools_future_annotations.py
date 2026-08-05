"""Tests for tools defined in a module using PEP 563 string annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from giskard.agents.context import RunContext
from giskard.agents.tools import tool
from pydantic import BaseModel

if TYPE_CHECKING:
    from decimal import Decimal


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

    result = await count_tool.run({"increment": 2}, ctx=context)

    # Pre-fix, RunContext detection silently fails and `context` leaks into
    # the model-facing schema as a required field. The resulting pydantic
    # validation error is then swallowed by the default `catch` handler and
    # returned as an ordinary (misleading) tool result, so pin the exact
    # output rather than only the side effect.
    assert context.get("call_count") == 2
    assert result == "2"


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


def test_type_checking_only_annotation_raises_name_error_at_decoration_time():
    with pytest.raises(NameError, match="Decimal"):

        @tool
        def price_tool(amount: Decimal) -> str:
            """Format a price.

            Parameters
            ----------
            amount : Decimal
                The amount to format.
            """
            return str(amount)
