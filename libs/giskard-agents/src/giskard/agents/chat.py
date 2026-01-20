from typing import Any, Generic, Literal, Type, TypeVar

from litellm import Message as LiteLLMMessage
from pydantic import BaseModel, Field

from .context import RunContext
from .errors.serializable import Error
from .tools import ToolCall

Role = Literal["assistant", "user", "system", "tool"]


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class File(BaseModel):
    data: bytes


class FileContent(BaseModel):
    type: Literal["file"] = "file"
    file: File


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


Content = TextContent | ThinkingContent | None


T = TypeVar("T", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class Message(BaseModel):
    role: Role
    content: str | Content | list[Content] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_litellm(self) -> dict[str, Any]:
        msg = self.model_dump(include={"role", "content", "tool_calls", "tool_call_id"})
        return msg

    @classmethod
    def from_litellm(cls, msg: LiteLLMMessage | dict[str, Any]):
        return cls.model_validate(msg.model_dump())  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    def parse(self, model_type: type[T]) -> T:
        return model_type.model_validate_json(self.content)  # pyright: ignore[reportArgumentType]

    @property
    def transcript(self) -> str:
        role = self.role
        if role == "tool" and self.tool_call_id is not None:
            role += f":{self.tool_call_id}"

        content = str(self.content)
        if self.tool_calls:
            for tool_call in self.tool_calls:
                content += f"\n>[tool_call:{tool_call.function.name}:{tool_call.id}]: {tool_call.function.arguments}"

        return f"[{role}]: {content}"


class Chat(BaseModel, Generic[OutputType]):
    messages: list[Message]
    output_model: Type[OutputType] | None = Field(default=None)
    context: RunContext = Field(default_factory=RunContext)

    error: Error | None = None

    @property
    def last(self) -> Message:
        return self.messages[-1]

    @property
    def transcript(self) -> str:
        return "\n".join([m.transcript for m in self.messages])

    @property
    def output(self) -> OutputType:
        if self.output_model is None:
            raise ValueError("Output model not set")
        return self.last.parse(self.output_model)

    @property
    def failed(self) -> bool:
        return self.error is not None

    def clone(
        self, deep: bool = True, preserve_context: bool = True
    ) -> "Chat[OutputType]":
        cloned = self.model_copy(deep=deep)
        if preserve_context:
            cloned.context = self.context
        return cloned

    def add(self, message: Message) -> "Chat[OutputType]":
        self.messages.append(message)
        return self
