from typing import Any, Literal, Required, TypedDict

# -- Response Input types -------------------------------------------------------------


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal["input_text"]]


class ResponseRefusalParam(TypedDict, total=False):
    type: Required[Literal["refusal"]]
    refusal: Required[str]


class ResponseOutputTextBlockParam(TypedDict, total=False):
    type: Required[Literal["output_text"]]
    text: Required[str]


class ResponseFunctionCallOutputParam(TypedDict, total=False):
    type: Required[Literal["function_call_output"]]
    call_id: Required[str]
    output: Required[str]
    name: str


FunctionCallOutputParam = ResponseFunctionCallOutputParam


class ResponseFunctionToolCallParam(TypedDict, total=False):
    type: Required[Literal["function_call"]]
    arguments: Required[dict[str, Any]]
    call_id: Required[str]
    name: Required[str]


ResponseInputMessageContentParam = ResponseInputTextParam

ResponseOutputMessageContentParam = ResponseRefusalParam | ResponseOutputTextBlockParam


class ResponseEasyInputMessageParam(TypedDict, total=False):
    type: Literal["message"]
    content: Required[str | list[ResponseInputMessageContentParam]]
    role: Required[Literal["user", "assistant", "system", "developer"]]


class ResponseOutputMessageParam(TypedDict, total=False):
    type: Literal["message"]
    content: Required[str | list[ResponseOutputMessageContentParam]]
    role: Required[Literal["assistant"]]


class ResponseReasoningSummaryParam(TypedDict, total=False):
    type: Required[Literal["summary_text"]]
    text: Required[str]


class ResponseReasoningTextParam(TypedDict, total=False):
    type: Required[Literal["reasoning_text"]]
    text: Required[str]


class ResponseReasoningItemParam(TypedDict, total=False):
    type: Required[Literal["reasoning"]]
    id: Required[str]
    summary: Required[list[ResponseReasoningSummaryParam]]
    content: list[ResponseReasoningTextParam]
    encrypted_content: str
    status: Literal["in_progress", "completed", "incomplete"]


ResponseInputItemParam = (
    ResponseFunctionCallOutputParam
    | ResponseFunctionToolCallParam
    | ResponseEasyInputMessageParam
    | ResponseOutputMessageParam
    | ResponseReasoningItemParam
)
