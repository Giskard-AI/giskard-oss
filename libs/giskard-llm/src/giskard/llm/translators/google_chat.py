import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from ..types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionContent,
    CompletionResponse,
    FunctionMessage,
    RefusalContent,
    TextContent,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    ToolMessage,
    Usage,
    UserMessage,
)
from ..types._base import _BaseModel
from ..types._serialization import register_serializer
from ..utils import deserialize_arguments

if TYPE_CHECKING:
    from google.genai.types import (
        ContentListUnionDict,
        ContentUnionDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        PartDict,
        ToolDict,
    )

    class GenerateContentParams(TypedDict, total=False):
        model: Required[str]
        contents: Required[ContentListUnionDict]
        config: GenerateContentConfigDict


PROVIDER = "google"

KNOWN_COMPLETION_PARAMS = frozenset(
    {"temperature", "max_tokens", "tools", "response_format", "safety_settings"}
)

logger = logging.getLogger(__name__)


@register_serializer(ToolDef, "google/chat")
def serialize_tool_def(tool: ToolDef, _info: SerializationInfo) -> "ToolDict":
    return {
        "function_declarations": [
            {
                "name": tool.function.name,
                "description": tool.function.description or "No description provided",
                "parameters": tool.function.parameters,  # pyright: ignore[reportReturnType]
            }
        ]
    }


def _text_content(text: str) -> "PartDict":
    return {"text": text}


@register_serializer(TextContent, "google/chat")
def serialize_text_content(
    content: TextContent, _info: SerializationInfo
) -> "PartDict":
    return _text_content(content.text)


@register_serializer(RefusalContent, "google/chat")
def serialize_refusal_content(
    content: RefusalContent, _info: SerializationInfo
) -> "PartDict":
    return _text_content(content.refusal)


def _assistant_content_to_parts(
    content: str | list[CompletionContent], info: SerializationInfo
) -> "Sequence[PartDict]":
    if isinstance(content, str):
        return [_text_content(content)]
    return [
        cast("PartDict", cast(object, c.model_dump(context=info.context)))
        for c in content
    ]


@register_serializer(ToolCall, "google/chat")
def serialize_tool_call(tool_call: ToolCall, info: SerializationInfo) -> "PartDict":
    return {
        "function_call": {
            "name": tool_call.function.name,
            "args": tool_call.function.arguments,
        }
    }


@register_serializer(UserMessage, "google/chat")
def serialize_user_message(
    message: UserMessage, info: SerializationInfo
) -> "ContentUnionDict":
    return {
        "role": "user",
        "parts": [_text_content(message.content)],
    }


def _get_name_from_call_id(call_id: str, info: SerializationInfo) -> str | None:
    if not isinstance(info.context, dict):
        return None

    call_ids_to_name = info.context.get("tool_call_id_to_name", {})

    if call_id in call_ids_to_name:
        return call_ids_to_name[call_id]
    return None


@register_serializer(ToolMessage, "google/chat")
def serialize_tool_message(
    message: ToolMessage, info: SerializationInfo
) -> "ContentUnionDict":
    name = _get_name_from_call_id(message.tool_call_id, info)
    if name is None:
        raise ValueError(f"Tool call id {message.tool_call_id} not found in context")

    return {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": name,
                    "response": {"result": message.content},
                }
            }
        ],
    }


@register_serializer(FunctionMessage, "google/chat")
def serialize_function_message(
    message: FunctionMessage, info: SerializationInfo
) -> "ContentUnionDict":
    raise ValueError(f"Unsupported message role for google chat: {message.role}")


@register_serializer(AssistantMessage, "google/chat")
def serialize_assistant_message(
    message: AssistantMessage, info: SerializationInfo
) -> "ContentUnionDict":
    parts = []
    if message.content is not None:
        parts.extend(_assistant_content_to_parts(message.content, info))
    if message.refusal is not None:
        parts.append(_text_content(message.refusal))
    if message.tool_calls is not None:
        parts.extend(
            [
                cast("PartDict", cast(object, tc.model_dump(context=info.context)))
                for tc in message.tool_calls
            ]
        )
    return {
        "role": "model",
        "parts": parts,
    }


def _extract_system_instruction(messages: Sequence[ChatMessage]) -> list[str] | None:
    system_parts = [
        m.content for m in messages if m.role == "system" or m.role == "developer"
    ]
    return system_parts if system_parts else None


class GoogleChatConfigParams(_BaseModel):
    tools: Sequence[ToolDef] | None
    system_instruction: str | list[str] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, validation_alias="max_tokens")
    response_mime_type: Literal["application/json"] | None = None
    response_schema: type[BaseModel] | None = None


class GoogleChatParams(_BaseModel):
    model: str
    contents: Sequence[ChatMessage]
    config: GoogleChatConfigParams

    @field_serializer("contents")
    def serialize_messages(
        self, value: Sequence[ChatMessage], info: SerializationInfo
    ) -> Any:
        tool_call_id_to_name: dict[str, str] = {}
        for m in value:
            if isinstance(m, AssistantMessage):
                for tc in m.tool_calls or []:
                    tool_call_id_to_name[tc.id] = tc.function.name

        if isinstance(info.context, dict):
            context = info.context.copy()
            context["tool_call_id_to_name"] = tool_call_id_to_name
        else:
            context = info.context

        return [
            cast("ChatMessage", cast(object, m.model_dump(context=context)))
            for m in value
        ]

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v

        v = v.copy()

        v["config"] = v.get("config", {})

        # Extract system instruction from messages
        system_instruction = _extract_system_instruction(v["contents"])
        if system_instruction:
            v["config"]["system_instruction"] = system_instruction

        # Remove system and developer messages from messages
        v["contents"] = [
            m for m in v["contents"] if m.role not in ("system", "developer")
        ]

        # Setup response_format for JSON output
        if (
            "response_format" in v["config"]
            and isinstance(v["config"]["response_format"], type)
            and issubclass(v["config"]["response_format"], BaseModel)
        ):
            v["config"]["response_mime_type"] = "application/json"
            v["config"]["response_schema"] = v["config"].pop("response_format")

        return v


class GoogleChatTranslator:
    @staticmethod
    def to_google(
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        **params: Any,
    ) -> "GenerateContentParams":
        unknown = set(params) - KNOWN_COMPLETION_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown completion params: %s",
                PROVIDER,
                sorted(unknown),
            )

        params_copy = dict(params)
        config_base = dict(params_copy.pop("config", {}))
        google_params = GoogleChatParams.model_validate(
            {
                "model": model,
                "contents": messages,
                "config": {
                    "tools": tools,
                    **config_base,
                    **params_copy,
                },
            }
        )

        return cast(
            "GenerateContentParams",
            cast(object, google_params.model_dump(context={"provider": "google/chat"})),
        )

    @staticmethod
    def from_google(raw: "GenerateContentResponse", model: str) -> CompletionResponse:
        choices: list[Choice] = []
        if not raw.candidates:
            return CompletionResponse(choices=[], model=model)

        for i, candidate in enumerate(raw.candidates):
            content = None
            tool_calls: list[ToolCall] | None = None
            finish_reason = "stop"

            if candidate.finish_reason:
                finish_reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                }
                finish_reason = finish_reason_map.get(
                    str(candidate.finish_reason), "stop"
                )

            if candidate.content and candidate.content.parts:
                text_parts = []
                fc_list: list[ToolCall] = []
                for part in candidate.content.parts:
                    if part.text is not None:
                        text_parts.append(part.text)
                    elif part.function_call is not None:
                        fc = part.function_call
                        raw_args = fc.args
                        if raw_args is None:
                            tool_args: dict[str, Any] = {}
                        elif isinstance(raw_args, (str, dict)):
                            tool_args = deserialize_arguments(raw_args)
                        else:
                            tool_args = {}
                        fc_list.append(
                            ToolCall(
                                id=f"call_{uuid4().hex[:8]}",
                                type="function",
                                function=ToolCallFunction(
                                    name=fc.name or "",
                                    arguments=tool_args,
                                ),
                            )
                        )
                content = "\n".join(text_parts) if text_parts else None
                if fc_list:
                    tool_calls = fc_list
                    finish_reason = "tool_calls"

            choices.append(
                Choice(
                    message=AssistantMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                    index=i,
                )
            )

        usage = None
        if raw.usage_metadata:
            usage = Usage(
                input_tokens=raw.usage_metadata.prompt_token_count or 0,
                output_tokens=raw.usage_metadata.candidates_token_count or 0,
                total_tokens=raw.usage_metadata.total_token_count or 0,
            )

        return CompletionResponse(choices=choices, model=model, usage=usage)
