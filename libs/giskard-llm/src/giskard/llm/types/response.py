from typing import Literal

from pydantic import Field

from ._base import ArgumentDict, _BaseModel

# -- Response Input types -------------------------------------------------------------


class ResponseInputText(_BaseModel):
    text: str
    type: Literal["input_text"] = "input_text"


class ResponseOutputRefusal(_BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str

    @property
    def output_text(self) -> str:
        """Return the refusal content, or None if there is no refusal."""
        return self.refusal


class ResponseOutputText(_BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str

    @property
    def output_text(self) -> str:
        """Return the text content, or None if there is no text."""
        return self.text


class ResponseFunctionCallOutput(_BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    name: str | None = None


FunctionCallOutput = ResponseFunctionCallOutput


class ResponseFunctionToolCall(_BaseModel):
    type: Literal["function_call"] = "function_call"
    arguments: ArgumentDict
    call_id: str
    name: str


ResponseInputMessageContent = ResponseInputText

ResponseOutputMessageContent = ResponseOutputRefusal | ResponseOutputText


class ResponseEasyInputMessage(_BaseModel):
    type: Literal["message"] | None = "message"
    content: str | list[ResponseInputMessageContent]
    role: Literal["user", "assistant", "system", "developer"]

    @property
    def text(self) -> str | None:
        """Return the text content, or None if there is no text."""
        if isinstance(self.content, str):
            return self.content

        texts = [o.text for o in self.content if isinstance(o, ResponseInputText)]

        return "\n".join(texts) if texts else None


class ResponseOutputMessage(_BaseModel):
    type: Literal["message"] | None = "message"
    content: str | list[ResponseOutputMessageContent]
    role: Literal["assistant"] = "assistant"

    @property
    def text(self) -> str | None:
        """Return the text content, or None if there is no text."""
        return self.output_text

    @property
    def output_text(self) -> str | None:
        """Concatenate all text outputs, or None if there are none."""
        if isinstance(self.content, str):
            return self.content

        texts = [o.output_text for o in self.content]

        return "\n".join(texts) if texts else None

    @property
    def refusal(self) -> str | None:
        """Return joined refusal segments, or None if there are no refusal parts."""
        if isinstance(self.content, str):
            return None
        refusals = [
            c.refusal for c in self.content if isinstance(c, ResponseOutputRefusal)
        ]
        return "\n".join(refusals) if refusals else None


class ResponseReasoningSummary(_BaseModel):
    """OpenAI Responses ``summary_text`` part on a ``reasoning`` item."""

    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningText(_BaseModel):
    """OpenAI Responses ``reasoning_text`` part on a ``reasoning`` item."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ResponseReasoningItem(_BaseModel):
    """OpenAI Responses ``reasoning`` output/input item.

    Provider adapters map native thinking (Gemini thought parts/steps, and
    OpenAI's own reasoning items) onto this shape. Visible answers stay on
    ``message`` / ``output_text`` items; this item is never concatenated into
    ``ResponseResult.output_text``.
    """

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponseReasoningSummary] = Field(default_factory=list)
    content: list[ResponseReasoningText] | None = None
    encrypted_content: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


ResponseInputItem = (
    ResponseFunctionCallOutput
    | ResponseFunctionToolCall
    | ResponseEasyInputMessage
    | ResponseOutputMessage
    | ResponseReasoningItem
)
