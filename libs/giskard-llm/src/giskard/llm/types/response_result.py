from typing import Literal

from pydantic import AliasChoices, Field

from ._base import ArgumentDict, _BaseModel
from .response import ResponseOutputMessage
from .usage import Usage

# -- Response / Interaction types (Responses API + Interactions API) -----------


class ResponseOutputFunctionCall(_BaseModel):
    type: Literal["function_call"] = "function_call"
    call_id: str | None = None
    name: str
    arguments: ArgumentDict


# Plain assignment (not `type` statement) so isinstance(x, ResponseOutputItem) works at runtime.
ResponseOutputItem = ResponseOutputMessage | ResponseOutputFunctionCall


class ResponseResult(_BaseModel):
    id: str
    outputs: list[ResponseOutputItem] = Field(
        validation_alias=AliasChoices("output", "outputs")
    )
    model: str | None = None
    usage: Usage | None = None

    @property
    def output_text(self) -> str | None:
        """Concatenate all text outputs, or None if there are none."""
        content = [
            o.output_text
            for o in self.outputs
            if isinstance(o, ResponseOutputMessage)
            if o.output_text is not None
        ]
        return "\n".join(content) if content else None

    @property
    def function_calls(self) -> list[ResponseOutputFunctionCall]:
        """Return all function-call outputs."""
        return [o for o in self.outputs if isinstance(o, ResponseOutputFunctionCall)]
