import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, cast

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

if TYPE_CHECKING:
    from google.genai.types import (
        ContentListUnionDict,
        ContentUnionDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        Part,
        PartDict,
        ToolDict,
    )

    class GenerateContentParams(TypedDict, total=False):
        model: Required[str]
        contents: Required[ContentListUnionDict]
        config: GenerateContentConfigDict


_PROVIDER = "google/chat"
PROVIDER = "google"

KNOWN_COMPLETION_PARAMS = frozenset(
    {"temperature", "max_tokens", "tools", "response_format", "safety_settings"}
)

# Sentinel that skips Gemini 3 thought-signature validation when we have no real
# signature for a tool call. https://ai.google.dev/gemini-api/docs/thought-signatures
_SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"

logger = logging.getLogger(__name__)


@ToolDef.register_serializer(_PROVIDER)
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


@TextContent.register_serializer(_PROVIDER)
def serialize_text_content(
    content: TextContent, _info: SerializationInfo
) -> "PartDict":
    part = _text_content(content.text)
    if content.thought_signature is not None:
        part["thought_signature"] = content.thought_signature
    return part


@RefusalContent.register_serializer(_PROVIDER)
def serialize_refusal_content(
    content: RefusalContent, _info: SerializationInfo
) -> "PartDict":
    return _text_content(content.refusal)


def _assistant_content_to_parts(
    content: str | Sequence[CompletionContent], info: SerializationInfo
) -> "Sequence[PartDict]":
    if isinstance(content, str):
        return [_text_content(content)]
    return [
        cast("PartDict", cast(object, c.model_dump(context=info.context)))
        for c in content
    ]


@ToolCall.register_serializer(_PROVIDER)
def serialize_tool_call(tool_call: ToolCall, info: SerializationInfo) -> "PartDict":
    return {
        "function_call": {
            "name": tool_call.function.name,
            "args": tool_call.function.arguments,
        },
        "thought_signature": tool_call.thought_signature or _SKIP_THOUGHT_SIGNATURE,
    }


@UserMessage.register_serializer(_PROVIDER)
def serialize_user_message(
    message: UserMessage, info: SerializationInfo
) -> "ContentUnionDict":
    if isinstance(message.content, str):
        return {
            "role": "user",
            "parts": [_text_content(message.content)],
        }
    else:
        return {
            "role": "user",
            "parts": [
                cast("PartDict", cast(object, c.model_dump(context=info.context)))
                for c in message.content
            ],
        }


def _get_name_from_call_id(call_id: str, info: SerializationInfo) -> str | None:
    if not isinstance(info.context, dict):
        return None

    call_ids_to_name = info.context.get("tool_call_id_to_name", {})

    if call_id in call_ids_to_name:
        return call_ids_to_name[call_id]
    return None


@ToolMessage.register_serializer(_PROVIDER)
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


@FunctionMessage.register_serializer(_PROVIDER)
def serialize_function_message(
    message: FunctionMessage, info: SerializationInfo
) -> "ContentUnionDict":
    raise ValueError(f"Unsupported message role for google chat: {message.role}")


@AssistantMessage.register_serializer(_PROVIDER)
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
        m.text
        for m in messages
        if (m.role == "system" or m.role == "developer") and m.text is not None
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


REFUSAL_REASONS = frozenset(
    {
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_PROHIBITED_CONTENT",
    }
)

FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
} | {reason: "refusal" for reason in REFUSAL_REASONS}


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
            cast(object, google_params.model_dump(context={"provider": _PROVIDER})),
        )

    @staticmethod
    def part_content_to_giskard(
        part: "Part", num_messages: int, part_index: int
    ) -> CompletionContent | ToolCall:
        if part.text is not None:
            return TextContent(text=part.text, thought_signature=part.thought_signature)
        if part.function_call is not None:
            fc = part.function_call
            return ToolCall(
                id=f"call_{num_messages}_{part_index}",
                type="function",
                function=ToolCallFunction(
                    name=fc.name or "",
                    arguments=fc.args or {},
                ),
                thought_signature=part.thought_signature,
            )
        raise ValueError(f"Unsupported part content type: {part}")

    @staticmethod
    def parts_to_giskard(
        parts: "Sequence[Part]",
        num_messages: int,
    ) -> tuple[Sequence[CompletionContent], Sequence[ToolCall]]:
        content_and_tool_calls = [
            GoogleChatTranslator.part_content_to_giskard(part, num_messages, part_index)
            for part_index, part in enumerate(parts)
        ]
        content = [
            content
            for content in content_and_tool_calls
            if not isinstance(content, ToolCall)
        ]
        tool_calls = [
            tool_call
            for tool_call in content_and_tool_calls
            if isinstance(tool_call, ToolCall)
        ]
        return content, tool_calls

    @staticmethod
    def from_google(
        raw: "GenerateContentResponse", model: str, num_messages: int
    ) -> CompletionResponse:
        choices: list[Choice] = []
        if not raw.candidates:
            return CompletionResponse(choices=[], model=model)

        # ``candidate.finish_reason`` is a ``FinishReason`` enum whose ``str()`` is
        # ``"FinishReason.SAFETY"`` (etc.), not the bare ``"SAFETY"`` the map is keyed
        # on -- so read ``.value``. The SDK is an optional dep imported by the caller,
        # so ``FinishReason`` is resolved lazily here (never at module load).
        from google.genai.types import FinishReason

        for i, candidate in enumerate(raw.candidates):
            finish_reason = "stop"

            fr = candidate.finish_reason
            raw_finish_reason = (
                fr.value if isinstance(fr, FinishReason) else str(fr or "")
            )
            if fr:
                finish_reason = FINISH_REASON_MAP.get(raw_finish_reason, "stop")

            refusal_out = (
                (candidate.finish_message or raw_finish_reason)
                if finish_reason == "refusal"
                else None
            )

            if candidate.content and candidate.content.parts:
                content, tool_calls = GoogleChatTranslator.parts_to_giskard(
                    candidate.content.parts,
                    num_messages,
                )
                if tool_calls:
                    finish_reason = "tool_calls"
            else:
                content = None
                tool_calls = None

            choices.append(
                Choice(
                    message=AssistantMessage(
                        role="assistant",
                        content=content if content else None,
                        refusal=refusal_out,
                        tool_calls=tool_calls if tool_calls else None,
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
