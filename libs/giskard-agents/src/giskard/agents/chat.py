from giskard.llm.types import AssistantMessage, ChatMessage, ChatMessageParam
from pydantic import BaseModel, Field, TypeAdapter

from .context import RunContext
from .errors.serializable import Error

_CHAT_MESSAGE_TYPE_ADAPTER = TypeAdapter(ChatMessage)


class Chat[OutputType: BaseModel](BaseModel):
    messages: list[ChatMessage]
    output_model: type[OutputType] | None = Field(default=None)
    context: RunContext = Field(default_factory=RunContext)

    error: Error | None = None

    @property
    def last(self) -> ChatMessage:
        return self.messages[-1]

    @property
    def transcript(self) -> str:
        return "\n".join([m.transcript for m in self.messages])

    @property
    def output(self) -> OutputType:
        if self.output_model is None:
            raise ValueError("Output model not set")

        last = self.last
        if not isinstance(last, AssistantMessage):
            raise ValueError("Last message is not an assistant message")

        output_text = last.text
        if output_text is None:
            raise ValueError("Last message has no output text")

        return self.output_model.model_validate_json(output_text)

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

    def add(self, message: ChatMessage | ChatMessageParam) -> "Chat[OutputType]":
        self.messages.append(_CHAT_MESSAGE_TYPE_ADAPTER.validate_python(message))
        return self
