from typing import Literal

from ._base import _BaseModel

# -- Tool definition types (input side) ---------------------------------------


class FunctionDef(_BaseModel):
    """Schema for a function tool definition."""

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None


class ToolDef(_BaseModel):
    """OpenAI-format tool definition accepted by all providers."""

    type: Literal["function"] = "function"
    function: FunctionDef
