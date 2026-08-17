from giskard.llm.types import AssistantMessage, ChatMessage, ChatMessageParam
from pydantic import BaseModel, Field, TypeAdapter

from .context import RunContext
from .errors.serializable import Error

_CHAT_MESSAGE_TYPE_ADAPTER = TypeAdapter(ChatMessage)


class Chat[OutputType: BaseModel](BaseModel):
    """Message history of a workflow run, plus its context and error state."""

    messages: list[ChatMessage]
    output_model: type[OutputType] | None = Field(default=None)
    context: RunContext = Field(default_factory=RunContext)

    error: Error | None = None

    @property
    def last(self) -> ChatMessage:
        """Return the most recent message."""
        return self.messages[-1]

    @property
    def transcript(self) -> str:
        """Return every message rendered as a newline-separated transcript."""
        return "\n".join([m.transcript for m in self.messages])

    @property
    def output(self) -> OutputType:
        """Return the last assistant message parsed as ``output_model``.

        Raises
        ------
        ValueError
            If no output model is set, or the last message holds no parsable text.
        """
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
        """Return whether the run ended with an error."""
        return self.error is not None

    def clone(
        self, deep: bool = True, preserve_context: bool = True
    ) -> "Chat[OutputType]":
        """Copy the chat.

        Parameters
        ----------
        deep : bool, default True
            Deep-copy the messages instead of sharing them.
        preserve_context : bool, default True
            Keep the original :class:`RunContext` instance instead of copying it.

        Returns
        -------
        Chat
            The copy.
        """
        cloned = self.model_copy(deep=deep)
        if preserve_context:
            cloned.context = self.context
        return cloned

    def add(self, message: ChatMessage | ChatMessageParam) -> "Chat[OutputType]":
        """Validate and append a message, returning self for chaining."""
        self.messages.append(_CHAT_MESSAGE_TYPE_ADAPTER.validate_python(message))
        return self
