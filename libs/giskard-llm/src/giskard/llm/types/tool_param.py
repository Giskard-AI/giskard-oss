from typing import Literal, Required, TypedDict

# -- Tool definition types (input side) ---------------------------------------


class FunctionDefParam(TypedDict):
    """Schema for a function tool definition."""

    name: Required[str]
    description: str
    parameters: dict[str, object]


class ToolDefParam(TypedDict):
    """OpenAI-format tool definition accepted by all providers."""

    type: Required[Literal["function"]]
    function: Required[FunctionDefParam]
