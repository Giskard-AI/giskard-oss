from collections.abc import Sequence
from typing import Any, Literal, Required, TypedDict

# -- Chat content types -------------------------------------------------------------


class TextContentParam(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]


class RefusalContentParam(TypedDict, total=False):
    type: Required[Literal["refusal"]]
    refusal: Required[str]


CompletionContentParam = TextContentParam | RefusalContentParam

# -- Reasoning details --------------------------------------------------------------


class ReasoningTextDetailParam(TypedDict, total=False):
    type: Required[Literal["reasoning.text"]]
    text: Required[str]
    signature: str | None
    id: str | None
    format: str
    index: int


class ReasoningSummaryDetailParam(TypedDict, total=False):
    type: Required[Literal["reasoning.summary"]]
    summary: Required[str]
    id: str | None
    format: str
    index: int


class ReasoningEncryptedDetailParam(TypedDict, total=False):
    type: Required[Literal["reasoning.encrypted"]]
    data: Required[str]
    id: str | None
    format: str
    index: int


ReasoningDetailParam = (
    ReasoningTextDetailParam
    | ReasoningSummaryDetailParam
    | ReasoningEncryptedDetailParam
)

# -- Chat Message types -------------------------------------------------------------


class ToolCallFunctionParam(TypedDict, total=False):
    name: Required[str]
    arguments: Required[dict[str, Any]]


class ToolCallParam(TypedDict, total=False):
    id: Required[str]
    type: Required[Literal["function"]]
    function: Required[ToolCallFunctionParam]


class SystemMessageParam(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[str]


class DeveloperMessageParam(TypedDict, total=False):
    role: Required[Literal["developer"]]
    content: Required[str | Sequence[CompletionContentParam]]


class UserMessageParam(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[str | Sequence[CompletionContentParam]]


class AssistantMessageParam(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: str | Sequence[CompletionContentParam]
    refusal: str
    tool_calls: list[ToolCallParam]
    reasoning: str
    reasoning_details: list[ReasoningDetailParam]


class ToolMessageParam(TypedDict, total=False):
    content: Required[str | Sequence[CompletionContentParam]]
    role: Required[Literal["tool"]]
    tool_call_id: Required[str]


class FunctionMessageParam(TypedDict, total=False):
    content: Required[str | None]
    name: Required[str]
    role: Required[Literal["function"]]


ChatMessageParam = (
    SystemMessageParam
    | DeveloperMessageParam
    | UserMessageParam
    | AssistantMessageParam
    | ToolMessageParam
    | FunctionMessageParam
)
