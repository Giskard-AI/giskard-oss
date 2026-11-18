from pydantic import AliasChoices, Field

from ._base import _BaseModel
from .reasoning import ReasoningContent
from .response import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from .usage import Usage

# -- Response / Interaction types (Responses API + Interactions API) -----------


# Plain assignment (not `type` statement) so isinstance(x, ResponseOutputItem) works at runtime.
ResponseOutputItem = (
    ResponseOutputMessage
    | ResponseFunctionToolCall
    | ResponseReasoningItem
    | ReasoningContent
)


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
    def function_calls(self) -> list[ResponseFunctionToolCall]:
        """Return all function-call outputs."""
        return [o for o in self.outputs if isinstance(o, ResponseFunctionToolCall)]

    @property
    def reasoning(self) -> list[ReasoningContent | ResponseReasoningItem]:
        """Return all reasoning outputs (never part of ``output_text``)."""
        # ``ResponseReasoningItem`` is accepted until the translators migrate.
        return [
            o
            for o in self.outputs
            if isinstance(o, ReasoningContent | ResponseReasoningItem)
        ]
