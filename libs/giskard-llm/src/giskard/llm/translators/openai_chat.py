import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ChatMessage,
    CompletionResponse,
    ToolDef,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
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
        return cast(
            "ChatCompletionToolUnionParam",
            cast(object, tool.model_dump(context={"json_arguments": True})),
        )

    @staticmethod
    def _tools_to_openai(
        tools: Sequence["ToolDef"],
    ) -> Sequence["ChatCompletionToolUnionParam"]:
        return [OpenAIChatTranslator._tool_def_to_openai(tool) for tool in tools]

    @staticmethod
    def _message_to_openai(message: ChatMessage) -> "ChatCompletionMessageParam":
        return cast(
            "ChatCompletionMessageParam",
            cast(object, message.model_dump(context={"json_arguments": True})),
        )

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
    def from_openai(
        raw: "ChatCompletion",
    ) -> "CompletionResponse":
        return CompletionResponse.model_validate(raw.model_dump())
