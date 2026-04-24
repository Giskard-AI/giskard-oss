from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Required, TypedDict

from pydantic import BaseModel

from ..types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionContent,
    CompletionResponse,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    Usage,
)
from ..utils import deserialize_arguments

if TYPE_CHECKING:
    import httpx
    from anthropic.types.message import Message
    from anthropic.types.message_param import MessageParam
    from anthropic.types.model_param import ModelParam
    from anthropic.types.output_config_param import OutputConfigParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.tool_union_param import ToolUnionParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam

    class CompletionCreateParams(TypedDict, total=False):
        messages: Required[Sequence[MessageParam]]
        model: Required[ModelParam]
        max_tokens: Required[int]
        tools: Sequence[ToolUnionParam]
        system: str | list[TextBlockParam]
        temperature: float
        timeout: float | httpx.Timeout | None
        output_config: OutputConfigParam


class AnthropicChatTranslator:
    @staticmethod
    def _tool_to_anthropic(tool: ToolDef) -> "ToolUnionParam":
        """Convert an OpenAI-format tool to Anthropic format."""
        func = tool.function
        return {
            "name": func.name,
            "description": func.description or "",
            "input_schema": func.parameters or {},
        }

    @staticmethod
    def _string_to_text_block(text: str) -> "TextBlockParam":
        """Convert a tool call to an Anthropic tool use block."""
        return {
            "type": "text",
            "text": text,
        }

    @staticmethod
    def _str_to_text_blocks[T](
        content: "str | Iterable[T]",
    ) -> "Iterable[TextBlockParam | T]":
        if isinstance(content, str):
            return [AnthropicChatTranslator._string_to_text_block(content)]

        return content

    @staticmethod
    def _completion_content_to_block(
        content: CompletionContent,
    ) -> "TextBlockParam":
        if content.type == "text":
            return AnthropicChatTranslator._string_to_text_block(content.text)

        if content.type == "refusal":
            return AnthropicChatTranslator._string_to_text_block(content.refusal)

        raise ValueError(f"Unsupported content type: {content.type!r}")

    @staticmethod
    def _completion_content_to_blocks(
        content: str | Sequence[CompletionContent],
    ) -> "Sequence[TextBlockParam]":
        if isinstance(content, str):
            return [AnthropicChatTranslator._string_to_text_block(content)]

        return [
            AnthropicChatTranslator._completion_content_to_block(p) for p in content
        ]

    @staticmethod
    def _message_to_anthropic(
        message: ChatMessage,
    ) -> "MessageParam | None":
        """Convert a chat message to an Anthropic message."""
        if message.role == "system" or message.role == "developer":
            # Folded into top-level ``system`` (Anthropic has no developer role on messages).
            return None

        if message.role == "function":
            raise ValueError(f"Unsupported message role: {message.role}")

        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }

        if message.role == "user":
            return {
                "role": "user",
                "content": message.content,
            }

        if message.role == "assistant":
            content = message.content
            blocks: list[TextBlockParam | ToolUseBlockParam] = []
            if content is not None:
                blocks.extend(
                    AnthropicChatTranslator._completion_content_to_blocks(content)
                )
            if (refusal := message.refusal) is not None:
                blocks.append(AnthropicChatTranslator._string_to_text_block(refusal))

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

    @staticmethod
    def _messages_to_anthropic(
        messages: Sequence[ChatMessage],
    ) -> "Sequence[MessageParam]":
        """Convert chat messages to Anthropic format, merging adjacent same-role turns."""
        anthropic_messages: list[MessageParam] = []

        for raw in messages:
            converted = AnthropicChatTranslator._message_to_anthropic(raw)
            if converted is None:
                continue

            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == converted["role"]
            ):
                prev = anthropic_messages[-1]
                prev_content = list(
                    AnthropicChatTranslator._str_to_text_blocks(prev["content"])
                )
                curr_content = list(
                    AnthropicChatTranslator._str_to_text_blocks(converted["content"])
                )
                anthropic_messages[-1] = {
                    **prev,
                    "content": prev_content + curr_content,
                }
            else:
                anthropic_messages.append(converted)

        return anthropic_messages

    @staticmethod
    def _extract_system_messages(messages: Sequence[ChatMessage]) -> list[str]:
        """Extract system and developer messages in order (both map to ``system``)."""
        return [
            m.content for m in messages if m.role == "system" or m.role == "developer"
        ]

    @staticmethod
    def _extract_system_messages_to_blocks(
        messages: Sequence[ChatMessage],
    ) -> "list[TextBlockParam]":
        """Extract system messages from a list of messages and convert them to blocks."""
        system_messages = AnthropicChatTranslator._extract_system_messages(messages)
        return [
            AnthropicChatTranslator._string_to_text_block(system_message)
            for system_message in system_messages
        ]

    @staticmethod
    def to_anthropic(
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        **params: Any,
    ) -> "CompletionCreateParams":
        completion_params: "CompletionCreateParams" = {
            "model": model,
            "messages": AnthropicChatTranslator._messages_to_anthropic(messages),
            "max_tokens": params.get("max_tokens", 4096),
        }

        if tools is not None:
            completion_params["tools"] = [
                AnthropicChatTranslator._tool_to_anthropic(t) for t in tools
            ]

        if system_blocks := AnthropicChatTranslator._extract_system_messages_to_blocks(
            messages
        ):
            completion_params["system"] = system_blocks

        if params.get("temperature") is not None:
            completion_params["temperature"] = params["temperature"]
        if params.get("timeout") is not None:
            completion_params["timeout"] = params["timeout"]

        response_format = params.get("response_format")
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                completion_params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            elif isinstance(response_format, dict):
                # Let anthropic validate the output config
                completion_params["output_config"] = response_format  # pyright: ignore[reportGeneralTypeIssues]
            else:
                raise ValueError(f"Unsupported response format: {response_format}")

        return completion_params

    @staticmethod
    def from_anthropic(
        raw: "Message",
    ) -> CompletionResponse:
        """Convert raw SDK response to CompletionResponse."""
        content_text: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=ToolCallFunction(
                            name=block.name,
                            arguments=deserialize_arguments(block.input),
                        ),
                    )
                )

        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
            "refusal": "stop",
        }
        finish_reason = (
            finish_reason_map.get(raw.stop_reason, "stop") if raw.stop_reason else None
        )

        refusal_out: str | None = None
        if raw.stop_reason == "refusal" and raw.stop_details is not None:
            refusal_out = raw.stop_details.explanation

        message = AssistantMessage(
            role="assistant",
            content="\n".join(content_text) if content_text else None,
            refusal=refusal_out,
            tool_calls=tool_calls or None,
        )

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            )

        return CompletionResponse(
            choices=[Choice(message=message, finish_reason=finish_reason, index=0)],
            model=raw.model,
            usage=usage,
        )
