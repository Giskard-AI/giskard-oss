from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, Protocol

from pydantic import Field, model_validator

from ._base import ArgumentDict, _BaseModel

# -- Utility functions -------------------------------------------------------------


class _TextContentProtocol(Protocol):
    text: str | None


class _TextualizableContentProtocol(Protocol):
    @property
    def text(self) -> str | None: ...


def _extract_text(
    content: (
        str | Sequence[_TextualizableContentProtocol | _TextContentProtocol] | None
    ),
) -> str | None:
    if isinstance(content, str) or content is None:
        return content

    texts = [c.text for c in content if c.text is not None]

    return "\n".join(texts) if texts else None


# -- Chat content types -------------------------------------------------------------


class TextContent(_BaseModel):
    type: Literal["text"] = "text"
    text: str
    thought_signature: bytes | None = Field(
        default=None, exclude=True, repr=False
    )  # Gemini-specific


class RefusalContent(_BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str

    @property
    def text(self) -> str:
        return self.refusal


CompletionContent = TextContent | RefusalContent

# -- Reasoning details --------------------------------------------------------------
# De facto OpenAI-compatible chat convention (OpenRouter / vLLM): structured, replayable
# reasoning that must be passed back unmodified on later assistant turns.


class _ReasoningDetailBase(_BaseModel):
    id: str | None = None
    format: str | None = "unknown"
    index: int | None = None


class ReasoningTextDetail(_ReasoningDetailBase):
    """Plaintext reasoning, optionally signed for verbatim replay."""

    type: Literal["reasoning.text"] = "reasoning.text"
    text: str
    signature: str | None = None


class ReasoningSummaryDetail(_ReasoningDetailBase):
    """Provider-generated summary of the reasoning."""

    type: Literal["reasoning.summary"] = "reasoning.summary"
    summary: str


class ReasoningEncryptedDetail(_ReasoningDetailBase):
    """Opaque (encrypted or redacted) reasoning payload."""

    type: Literal["reasoning.encrypted"] = "reasoning.encrypted"
    data: str


ReasoningDetail = Annotated[
    ReasoningTextDetail | ReasoningSummaryDetail | ReasoningEncryptedDetail,
    Field(discriminator="type"),
]

# -- Chat Message types -------------------------------------------------------------


class ToolCallFunction(_BaseModel):
    name: str
    arguments: ArgumentDict


class ToolCall(_BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: ToolCallFunction
    thought_signature: bytes | None = Field(
        default=None, exclude=True, repr=False
    )  # Gemini-specific


class SystemMessage(_BaseModel):
    role: Literal["system"] = "system"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"


class DeveloperMessage(_BaseModel):
    role: Literal["developer"] = "developer"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"


class UserMessage(_BaseModel):
    role: Literal["user"] = "user"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"


class AssistantMessage(_BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | Sequence[CompletionContent] | None = None
    refusal: str | None = None
    tool_calls: Sequence[ToolCall] | None = None
    # Reasoning never contributes to ``text`` / ``transcript``: those are what checks
    # and judges evaluate, and must only contain the visible answer. It is only replayed
    # to the provider that produced it: the Anthropic serializer sends signed thinking
    # back (required for tool loops with thinking on). Every other serializer strips it
    # through ``_wire_excluded``.
    reasoning: str | None = None
    reasoning_details: list[ReasoningDetail] | None = None

    _wire_excluded: ClassVar[frozenset[str]] = frozenset(
        {"reasoning", "reasoning_details"}
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_reasoning_content(cls, data: Any) -> Any:
        # vLLM / DeepSeek name it ``reasoning_content``; a null ``reasoning`` key must
        # not shadow it (a plain ``AliasChoices`` picks the first key present).
        if isinstance(data, dict) and data.get("reasoning") is None:
            if (reasoning_content := data.get("reasoning_content")) is not None:
                return {**data, "reasoning": reasoning_content}
        return data

    @property
    def is_refusal(self) -> bool:
        if self.refusal is not None:
            return True

        if self.content is not None and not isinstance(self.content, str):
            return any(isinstance(c, RefusalContent) for c in self.content or [])

        return False

    @property
    def text(self) -> str | None:
        texts = [
            text
            for text in (self.refusal, _extract_text(self.content))
            if text is not None
        ]

        return "\n".join(texts) if texts else None

    @property
    def transcript(self) -> str:
        message = self.text or "empty"
        if self.tool_calls is not None:
            for tool_call in self.tool_calls:
                message += f"\n>[tool_call:{tool_call.function.name}:{tool_call.id}]: {tool_call.function.arguments}"

        return f"[{self.role}]: {message}"


class ToolMessage(_BaseModel):
    role: Literal["tool"] = "tool"
    content: str | Sequence[TextContent]
    tool_call_id: str

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"


class FunctionMessage(_BaseModel):
    content: str | None = None
    name: str
    role: Literal["function"] = "function"

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"


ChatMessage = (
    SystemMessage
    | DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | FunctionMessage
)
