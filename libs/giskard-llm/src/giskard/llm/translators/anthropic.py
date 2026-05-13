from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, cast

from pydantic import BaseModel, SerializationInfo, field_serializer, model_validator

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
)
from ..types._base import _BaseModel
from ..utils import deserialize_arguments

if TYPE_CHECKING:
    from anthropic.types.content_block import ContentBlock
    from anthropic.types.message import Message
    from anthropic.types.message_param import MessageParam
    from anthropic.types.model_param import ModelParam
    from anthropic.types.output_config_param import OutputConfigParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.tool_union_param import ToolUnionParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam
    from httpx import Timeout as httpxTimeout

    class CompletionCreateParams(TypedDict, total=False):
        messages: Required[Sequence[MessageParam]]
        model: Required[ModelParam]
        max_tokens: Required[int]
        tools: Sequence[ToolUnionParam]
        system: str | list[TextBlockParam]
        temperature: float
        timeout: float | httpxTimeout | None
        output_config: OutputConfigParam
else:
    httpxTimeout = Any

_PROVIDER = "anthropic/chat"


@ToolDef.register_serializer(_PROVIDER)
def serialize_tool_def(tool: ToolDef, _info: SerializationInfo) -> "ToolUnionParam":
    return {
        "name": tool.function.name,
        "description": tool.function.description or "",
        "input_schema": tool.function.parameters or {},
    }


def _text_block(text: str) -> "TextBlockParam":
    return {
        "type": "text",
        "text": text,
    }


@RefusalContent.register_serializer(_PROVIDER)
def serialize_text_content(
    content: RefusalContent, _info: SerializationInfo
) -> "TextBlockParam":
    return _text_block(content.refusal)


@ToolMessage.register_serializer(_PROVIDER)
def serialize_tool_message(
    message: ToolMessage, info: SerializationInfo
) -> "MessageParam":
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": _completion_content_to_blocks(message.content, info),
            }
        ],
    }


@FunctionMessage.register_serializer(_PROVIDER)
def serialize_function_message(
    message: FunctionMessage, info: SerializationInfo
) -> "MessageParam":
    raise ValueError(f"Unsupported message role for anthropic chat: {message.role}")


def _completion_content_to_blocks(
    content: str | Sequence[CompletionContent],
    info: SerializationInfo | None = None,
) -> "Sequence[TextBlockParam]":
    if isinstance(content, str):
        return [_text_block(content)]

    return [
        cast(
            "TextBlockParam",
            cast(object, c.model_dump(context=info.context if info else None)),
        )
        for c in content
    ]


@AssistantMessage.register_serializer(_PROVIDER)
def serialize_assistant_message(
    message: AssistantMessage, info: SerializationInfo
) -> "MessageParam":
    blocks: "list[TextBlockParam | ToolUseBlockParam]" = []
    if message.content is not None:
        blocks.extend(_completion_content_to_blocks(message.content, info))
    if (refusal := message.refusal) is not None:
        blocks.append(_text_block(refusal))

    if tool_calls := message.tool_calls:
        for tool_call in tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": tool_call.function.arguments,
                }
            )

    return {
        "role": "assistant",
        "content": blocks,
    }


def _str_to_text_blocks[T](
    content: "str | Iterable[T]",
) -> "list[TextBlockParam | T]":
    if isinstance(content, str):
        return [_text_block(content)]

    return list(content)


def _extract_system_instruction(
    messages: Sequence[ChatMessage],
) -> "list[TextBlockParam] | None":
    system_blocks = [
        block
        for m in messages
        if m.role == "system" or m.role == "developer"
        for block in _completion_content_to_blocks(m.content)
    ]
    return system_blocks if system_blocks else None


class SystemTextBlock(_BaseModel):
    text: str
    type: Literal["text"] = "text"


class AnthropicChatConfigParams(_BaseModel):
    model: str
    messages: Sequence[ChatMessage]
    max_tokens: int = 4096
    tools: Sequence[ToolDef] | None = None
    system: str | list[SystemTextBlock] | None = None
    temperature: float | None = None
    timeout: float | httpxTimeout | None = None
    output_config: dict[str, object] | None = None

    @field_serializer("messages")
    def serialize_messages(
        self, values: Sequence[ChatMessage], info: SerializationInfo
    ) -> Any:
        messages = [m.model_dump(context=info.context) for m in values]

        merged_messages: list[dict[str, Any]] = []
        for message in messages:
            last_message = merged_messages[-1] if merged_messages else None
            if not last_message or last_message["role"] != message["role"]:
                merged_messages.append(message)
                continue

            prev_content = _str_to_text_blocks(last_message["content"])
            curr_content = _str_to_text_blocks(message["content"])
            merged_messages[-1] = {
                **last_message,
                "content": prev_content + curr_content,
            }

        return merged_messages

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v

        v = v.copy()

        # Extract system instruction from messages
        system = _extract_system_instruction(v["messages"])
        if system:
            v["system"] = system

        # Remove system and developer messages from messages
        v["messages"] = [
            m for m in v["messages"] if m.role not in ("system", "developer")
        ]

        # Setup response_format for JSON output
        if "response_format" in v:
            if isinstance(v["response_format"], type) and issubclass(
                v["response_format"], BaseModel
            ):
                schema = v["response_format"].model_json_schema()
                schema["additionalProperties"] = False
                v["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            else:
                v["output_config"] = v["response_format"]

        return v


FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "stop",
}


class AnthropicChatTranslator:
    @staticmethod
    def to_anthropic(
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        **params: Any,
    ) -> "CompletionCreateParams":
        anthropic_params = AnthropicChatConfigParams(
            model=model,
            messages=messages,
            tools=tools,
            **params,
        )

        return cast(
            "CompletionCreateParams",
            cast(
                object,
                anthropic_params.model_dump(context={"provider": _PROVIDER}),
            ),
        )

    @staticmethod
    def block_content_to_giskard(
        block: "ContentBlock",
    ) -> CompletionContent | ToolCall:
        if block.type == "text":
            return TextContent(text=block.text)
        elif block.type == "tool_use":
            return ToolCall(
                id=block.id,
                type="function",
                function=ToolCallFunction(
                    name=block.name,
                    arguments=deserialize_arguments(block.input),
                ),
            )
        else:
            raise ValueError(f"Unsupported content block type: {block.type}")

    @staticmethod
    def blocks_to_giskard(
        blocks: "Sequence[ContentBlock]",
    ) -> tuple[Sequence[CompletionContent], Sequence[ToolCall]]:
        content_and_tool_calls = [
            AnthropicChatTranslator.block_content_to_giskard(block) for block in blocks
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
    def from_anthropic(
        raw: "Message",
    ) -> CompletionResponse:
        """Convert raw SDK response to CompletionResponse."""
        content, tool_calls = AnthropicChatTranslator.blocks_to_giskard(raw.content)

        finish_reason = (
            FINISH_REASON_MAP.get(raw.stop_reason, "stop") if raw.stop_reason else None
        )

        refusal_out: str | None = None
        if raw.stop_reason == "refusal" and raw.stop_details is not None:
            # stop_details.explanation is a beta Anthropic API attribute; use getattr for safety
            refusal_out = getattr(raw.stop_details, "explanation", None)

        message = AssistantMessage(
            role="assistant",
            content=content if content else None,
            refusal=refusal_out,
            tool_calls=tool_calls or None,
        )

        usage = None
        if raw.usage:
            usage = Usage(
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            )

        return CompletionResponse(
            choices=[Choice(message=message, finish_reason=finish_reason, index=0)],
            model=raw.model,
            usage=usage,
        )
