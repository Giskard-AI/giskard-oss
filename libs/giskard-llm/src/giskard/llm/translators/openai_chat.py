import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
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
from pydantic import BaseModel

from ..utils import deserialize_arguments, serialize_arguments

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_assistant_message_param import (
        ChatCompletionAssistantMessageParam,
    )
    from openai.types.chat.chat_completion_content_part_refusal_param import (
        ChatCompletionContentPartRefusalParam,
    )
    from openai.types.chat.chat_completion_content_part_text_param import (
        ChatCompletionContentPartTextParam,
    )
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCallUnion,
    )
    from openai.types.chat.chat_completion_message_tool_call_union_param import (
        ChatCompletionMessageToolCallUnionParam,
    )
    from openai.types.chat.chat_completion_tool_union_param import (
        ChatCompletionToolUnionParam,
    )
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
        ResponseFormat,
    )

    class CompletionCreateParamsWithTimeout(
        CompletionCreateParamsNonStreaming, total=False
    ):
        timeout: float | int | None


logger = logging.getLogger(__name__)

PROVIDER = "openai"

KNOWN_COMPLETION_PARAMS = frozenset(
    {"temperature", "max_tokens", "timeout", "tools", "response_format", "metadata"}
)


class OpenAIChatTranslator:
    @staticmethod
    def _tool_def_to_openai(tool: "ToolDef") -> "ChatCompletionToolUnionParam":
        return {
            "type": "function",
            "function": {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "parameters": tool.function.parameters or {},
                "strict": None,
            },
        }

    @staticmethod
    def _tools_to_openai(
        tools: Sequence["ToolDef"],
    ) -> Sequence["ChatCompletionToolUnionParam"]:
        return [OpenAIChatTranslator._tool_def_to_openai(tool) for tool in tools]

    @staticmethod
    def _tool_call_to_openai(
        tool_call: "ToolCall",
    ) -> "ChatCompletionMessageToolCallUnionParam":
        return {
            "type": "function",
            "id": tool_call.id,
            "function": {
                "name": tool_call.function.name,
                "arguments": serialize_arguments(tool_call.function.arguments),
            },
        }

    @staticmethod
    def _completion_content_to_openai(
        content: CompletionContent,
    ) -> "ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam":
        if content.type == "text":
            return {"type": "text", "text": content.text}

        if content.type == "refusal":
            return {"type": "refusal", "refusal": content.refusal}

        raise ValueError(f"Unsupported content type: {content.type!r}")

    @staticmethod
    def _content_to_openai(
        content: str | list[CompletionContent],
    ) -> "str | Sequence[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam]":
        if isinstance(content, str):
            return content

        return [OpenAIChatTranslator._completion_content_to_openai(c) for c in content]

    @staticmethod
    def _message_to_openai(message: ChatMessage) -> "ChatCompletionMessageParam":
        if message.role == "system":
            return {
                "role": "system",
                "content": message.content,
            }

        if message.role == "developer":
            return {
                "role": "developer",
                "content": message.content,
            }

        if message.role == "user":
            return {
                "role": "user",
                "content": message.content,
            }

        if message.role == "assistant":
            assistant_msg: "ChatCompletionAssistantMessageParam" = {
                "role": "assistant",
            }
            if (content := message.content) is not None:
                assistant_msg["content"] = OpenAIChatTranslator._content_to_openai(
                    content
                )
            if (refusal := message.refusal) is not None:
                assistant_msg["refusal"] = refusal

            tool_calls = message.tool_calls
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    OpenAIChatTranslator._tool_call_to_openai(tool_call)
                    for tool_call in tool_calls
                ]

            return assistant_msg

        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }

        if message.role == "function":
            return {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }

        raise ValueError(f"Unsupported message role: {message.role!r}")

    @staticmethod
    def messages_to_openai(
        messages: Sequence[ChatMessage],
    ) -> list["ChatCompletionMessageParam"]:
        return [
            OpenAIChatTranslator._message_to_openai(message) for message in messages
        ]

    @staticmethod
    def to_openai(
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        **params: Any,
    ) -> "CompletionCreateParamsWithTimeout":
        unknown = set(params) - KNOWN_COMPLETION_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown completion params: %s",
                PROVIDER,
                sorted(unknown),
            )

        completion_params: "CompletionCreateParamsWithTimeout" = {
            "model": model,
            "messages": OpenAIChatTranslator.messages_to_openai(messages),
        }

        if tools is not None:
            completion_params["tools"] = OpenAIChatTranslator._tools_to_openai(tools)

        if params.get("temperature") is not None:
            completion_params["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            completion_params["max_tokens"] = params["max_tokens"]
        if params.get("timeout") is not None:
            completion_params["timeout"] = params["timeout"]
        if metadata := params.get("metadata"):
            completion_params["metadata"] = metadata

        response_format = params.get("response_format")
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }

            completion_params["response_format"] = cast(
                "ResponseFormat", response_format
            )

        return completion_params

    @staticmethod
    def _tool_call_from_openai(
        tool_call: "ChatCompletionMessageToolCallUnion",
    ) -> ToolCall:
        if tool_call.type == "function":
            return ToolCall(
                id=tool_call.id,
                type=tool_call.type,
                function=ToolCallFunction(
                    name=tool_call.function.name,
                    arguments=deserialize_arguments(tool_call.function.arguments),
                ),
            )

        raise ValueError(f"Unknown tool call type: {tool_call.type}")

    @staticmethod
    def _tool_calls_from_openai(
        tool_calls: Sequence["ChatCompletionMessageToolCallUnion"],
    ) -> Sequence[ToolCall]:
        return [
            OpenAIChatTranslator._tool_call_from_openai(tool_call)
            for tool_call in tool_calls
        ]

    @staticmethod
    def from_openai(
        raw: "ChatCompletion",
    ) -> "CompletionResponse":
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        role=c.message.role,
                        content=c.message.content,
                        refusal=c.message.refusal,
                        tool_calls=[
                            OpenAIChatTranslator._tool_call_from_openai(tc)
                            for tc in c.message.tool_calls
                        ]
                        if c.message.tool_calls
                        else None,
                    ),
                    finish_reason=c.finish_reason,
                    index=c.index,
                )
                for c in raw.choices
            ],
            model=raw.model,
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            )
            if raw.usage
            else None,
        )
