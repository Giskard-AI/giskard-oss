from typing import Literal

from ._base import ArgumentDict, _BaseModel

# -- Chat content types -------------------------------------------------------------


class TextContent(_BaseModel):
    type: Literal["text"] = "text"
    text: str

    @property
    def output_text(self) -> str:
        return self.text


class RefusalContent(_BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str

    @property
    def output_text(self) -> str:
        return self.refusal


CompletionContent = TextContent | RefusalContent

# -- Chat Message types -------------------------------------------------------------


class ToolCallFunction(_BaseModel):
    name: str
    arguments: ArgumentDict


class ToolCall(_BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: ToolCallFunction


class SystemMessage(_BaseModel):
    role: Literal["system"] = "system"
    content: str

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.content}"


class DeveloperMessage(_BaseModel):
    role: Literal["developer"] = "developer"
    content: str

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.content}"


class UserMessage(_BaseModel):
    role: Literal["user"] = "user"
    content: str

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.content}"


class AssistantMessage(_BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | list[CompletionContent] | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def output_text(self) -> str | None:
        if self.content is None:
            return self.refusal

        if isinstance(self.content, str):
            return self.content

        return (
            "\n".join([c.output_text for c in self.content]) if self.content else None
        )

    @property
    def transcript(self) -> str:
        message = f"[{self.role}]: {self.output_text}:"
        if self.tool_calls is not None:
            for tool_call in self.tool_calls:
                message += f"\n>[tool_call:{tool_call.function.name}:{tool_call.id}]: {tool_call.function.arguments}"

        return f"[{self.role}]: {message}"


class ToolMessage(_BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.content}"


class FunctionMessage(_BaseModel):
    content: str | None = "None"
    name: str
    role: Literal["function"] = "function"

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.content}"


ChatMessage = (
    SystemMessage
    | DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | FunctionMessage
)
