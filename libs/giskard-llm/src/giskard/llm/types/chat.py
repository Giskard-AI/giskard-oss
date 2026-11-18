from collections.abc import Sequence
from typing import Any, ClassVar, Literal, Protocol

from pydantic import Field, model_validator

from ._base import ArgumentDict, _BaseModel
from .reasoning import ReasoningContent

# -- Utility functions -------------------------------------------------------------


class _TextContentProtocol(Protocol):
    text: str | None


class _TextualizableContentProtocol(Protocol):
    @property
    def text(self) -> str | None: ...


def _extract_text(
    content: (
        str
        | Sequence[
            _TextualizableContentProtocol | _TextContentProtocol | ReasoningContent
        ]
        | None
    ),
) -> str | None:
    if isinstance(content, str) or content is None:
        return content

    # Reasoning is skipped by type: it must never leak into answer text.
    texts = [
        c.text
        for c in content
        if not isinstance(c, ReasoningContent) and c.text is not None
    ]

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


CompletionContent = TextContent | RefusalContent | ReasoningContent

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
    # and judges evaluate, and must only contain the visible answer. This field is a
    # deprecated, derived view of the ``ReasoningContent`` parts and is never replayed
    # (``_wire_excluded``); replay is driven by those parts, each sent back only to
    # the provider that produced it (ADR-0003).
    reasoning: str | None = Field(
        default=None,
        deprecated=(
            "`AssistantMessage.reasoning` is deprecated and will be removed in 2.0; "
            "read the `ReasoningContent` parts of `content` instead."
        ),
    )
    """Deprecated: plain text of the reasoning parts in ``content``.

    Filled from their ``reasoning_text`` when not given explicitly. Never
    replayed to a provider. Removed in 2.0; read ``ReasoningContent`` parts of
    ``content`` instead.
    """

    _wire_excluded: ClassVar[frozenset[str]] = frozenset({"reasoning"})

    @model_validator(mode="before")
    @classmethod
    def _accept_reasoning_content(cls, data: Any) -> Any:
        # vLLM / DeepSeek name it ``reasoning_content``; a null ``reasoning`` key must
        # not shadow it (a plain ``AliasChoices`` picks the first key present).
        if isinstance(data, dict) and data.get("reasoning") is None:
            if (reasoning_content := data.get("reasoning_content")) is not None:
                return {**data, "reasoning": reasoning_content}
        return data

    @model_validator(mode="after")
    def _derive_reasoning_from_parts(self) -> "AssistantMessage":
        # Read through ``__dict__``: the field is deprecated and attribute reads
        # would warn inside giskard-llm itself (assignment does not warn).
        if self.__dict__.get("reasoning") is not None or isinstance(
            self.content, str | None
        ):
            return self
        texts = [
            text.text
            for part in self.content
            if isinstance(part, ReasoningContent)
            for text in part.content or []
            if text.text
        ]
        if texts:
            self.reasoning = "\n".join(texts)
        return self

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
