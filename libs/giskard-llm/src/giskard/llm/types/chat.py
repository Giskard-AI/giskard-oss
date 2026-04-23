from typing import Any, Literal

from ._base import _BaseModel

# -- Chat content types -------------------------------------------------------------


class TextContent(_BaseModel):
    type: Literal["text"] = "text"
    text: str


class RefusalContent(_BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


CompletionContent = TextContent | RefusalContent

# -- Chat Message types -------------------------------------------------------------


class ToolCallFunction(_BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(_BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: ToolCallFunction


class SystemMessage(_BaseModel):
    role: Literal["system"] = "system"
    content: str


class DeveloperMessage(_BaseModel):
    role: Literal["developer"] = "developer"
    content: str


class UserMessage(_BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(_BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | list[CompletionContent] | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(_BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


class FunctionMessage(_BaseModel):
    content: str | None = "None"
    name: str
    role: Literal["function"] = "function"


type ChatMessage = (
    SystemMessage
    | DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | FunctionMessage
)
