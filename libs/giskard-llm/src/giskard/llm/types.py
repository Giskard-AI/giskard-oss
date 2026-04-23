"""Response types for giskard-llm.

These mirror the OpenAI-style response shapes that litellm used,
so existing code in giskard-agents can consume them with minimal changes.
"""

from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel, Field


class _BaseModel(BaseModel):
    """Shared base for all giskard-llm response models. Defaults model_dump to exclude None fields."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# -- Tool definition types (input side) ---------------------------------------


class FunctionDef(TypedDict):
    """Schema for a function tool definition."""

    name: Required[str]
    description: str
    parameters: dict[str, object]


class ToolDef(TypedDict):
    """OpenAI-format tool definition accepted by all providers."""

    type: Required[Literal["function"]]
    function: Required[FunctionDef]


class FunctionCallOutput(TypedDict):
    """Canonical format for feeding back a tool result to respond()."""

    type: Literal["function_call_output"]
    call_id: str
    name: str
    output: dict[str, Any]


# -- Tool call types (output side) --------------------------------------------


class ToolCallFunction(_BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(_BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


# -- Chat Completion types -----------------------------------------------------


class ChoiceMessage(_BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(_BaseModel):
    message: ChoiceMessage
    finish_reason: str | None = None
    index: int = 0


class Usage(_BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(_BaseModel):
    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None


# -- Embedding types -----------------------------------------------------------


class EmbeddingData(_BaseModel):
    embedding: list[float]
    index: int = 0


class EmbeddingUsage(_BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(_BaseModel):
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: EmbeddingUsage | None = None


# -- Response / Interaction types (Responses API + Interactions API) -----------


class ResponseOutputText(_BaseModel):
    type: Literal["text"] = "text"
    text: str


class ResponseOutputFunctionCall(_BaseModel):
    type: Literal["function_call"] = "function_call"
    call_id: str | None = None
    name: str
    arguments: dict[str, Any]


# Plain assignment (not `type` statement) so isinstance(x, ResponseOutputItem) works at runtime.
ResponseOutputItem = ResponseOutputText | ResponseOutputFunctionCall


class ResponseResult(_BaseModel):
    id: str
    outputs: list[ResponseOutputItem]
    model: str | None = None
    usage: Usage | None = None

    @property
    def output_text(self) -> str | None:
        """Concatenate all text outputs, or None if there are none."""
        texts = [o.text for o in self.outputs if isinstance(o, ResponseOutputText)]
        return "\n".join(texts) if texts else None

    @property
    def function_calls(self) -> list[ResponseOutputFunctionCall]:
        """Return all function-call outputs."""
        return [o for o in self.outputs if isinstance(o, ResponseOutputFunctionCall)]


# -- Chat Message types -------------------------------------------------------------


class ToolCallFunctionDict(TypedDict, total=False):
    name: Required[str]
    arguments: Required[dict[str, Any]]


class ToolCallDict(TypedDict, total=False):
    id: Required[str]
    type: Required[Literal["function"]]
    function: Required[ToolCallFunctionDict]


class SystemMessage(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[str]


class DeveloperMessage(TypedDict, total=False):
    role: Required[Literal["developer"]]
    content: Required[str]


class UserMessage(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[str]


class AssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: str
    tool_calls: list[ToolCallDict]


class ToolMessage(TypedDict, total=False):
    content: Required[str]
    role: Required[Literal["tool"]]
    tool_call_id: Required[str]


class FunctionMessage(TypedDict, total=False):
    content: Required[str | None]
    name: Required[str]
    role: Required[Literal["function"]]


ChatMessage = (
    SystemMessage
    | DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | FunctionMessage
)

# -- Response Input types -------------------------------------------------------------


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal["input_text"]]


class ResponseFunctionCallOutput(TypedDict, total=False):
    type: Required[Literal["function_call_output"]]
    call_id: Required[str]
    output: Required[str]
    id: str | None


class ResponseFunctionToolCall(TypedDict, total=False):
    type: Required[Literal["function_call"]]
    arguments: Required[dict[str, Any]]
    call_id: Required[str]
    name: Required[str]
    id: str | None


ResponseInputMessageContent = ResponseInputTextParam
ResponseInputMessageContentList = list[ResponseInputMessageContent]


class ResponseEasyInputMessage(TypedDict, total=False):
    type: Literal["message"]
    content: Required[str | ResponseInputMessageContentList]
    role: Required[Literal["user", "assistant", "system", "developer"]]


ResponseInputItem = (
    ResponseFunctionCallOutput | ResponseFunctionToolCall | ResponseEasyInputMessage
)
