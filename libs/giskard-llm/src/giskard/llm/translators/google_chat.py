import logging
from collections.abc import Sequence
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
from .anthropic import AnthropicChatTranslator

if TYPE_CHECKING:
    from google.genai.types import (
        ContentListUnionDict,
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


class GoogleChatTranslator:
    @staticmethod
    def _tool_to_google(tool: ToolDef) -> "ToolDict":
        """Convert an OpenAI-format tool to Gemini FunctionDeclaration."""
        func = tool.function
        return {
            "function_declarations": [
                {
                    "name": func.name,
                    "description": func.description,
                    "parameters": func.parameters,  # pyright: ignore[reportReturnType]
                }
            ]
        }

    @staticmethod
    def _content_to_parts(content: str) -> "PartDict":
        return {"text": content}

    @staticmethod
    def _completion_content_to_parts(content: CompletionContent) -> "PartDict":
        if content.type == "text":
            return {"text": content.text}

        if content.type == "refusal":
            return {"text": content.refusal}

        raise ValueError(f"Unsupported content type: {content.type!r}")

    @staticmethod
    def _assistant_content_to_parts(
        content: str | list[CompletionContent],
    ) -> "Sequence[PartDict]":
        if isinstance(content, str):
            return [GoogleChatTranslator._content_to_parts(content)]

        return [GoogleChatTranslator._completion_content_to_parts(p) for p in content]

    @staticmethod
    def _message_to_contents(
        message: ChatMessage, tc_id_to_name: dict[str, str]
    ) -> "ContentListUnionDict | None":
        if message.role == "system" or message.role == "developer":
            # Folded into ``system_instruction`` (Gemini has no developer turn in ``contents``).
            return None

        if message.role == "function":
            raise ValueError(f"Unsupported message role: {message.role}")

        if message.role == "user":
            return {
                "role": "user",
                "parts": [GoogleChatTranslator._content_to_parts(message.content)],
            }

        if message.role == "assistant":
            parts = []

            if message.content is not None:
                parts.extend(
                    GoogleChatTranslator._assistant_content_to_parts(message.content)
                )
            if message.refusal is not None:
                parts.append({"text": message.refusal})

            if tool_calls := message.tool_calls:
                for tc in tool_calls:
                    func = tc.function
                    parts.append(
                        {
                            "function_call": {
                                "name": func.name,
                                "args": func.arguments,
                            }
                        }
                    )

            return {
                "role": "model",
                "parts": parts,
            }

        if message.role == "tool":
            tc_id = message.tool_call_id
            return {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": tc_id_to_name.get(tc_id, tc_id),
                            "response": {"result": message.content},
                        }
                    }
                ],
            }

    @staticmethod
    def _messages_to_contents(
        messages: Sequence[ChatMessage],
    ) -> "ContentListUnionDict":
        tc_id_to_name: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for tc in msg.tool_calls or []:
                    tc_id_to_name[tc.id] = tc.function.name

        converted = [
            GoogleChatTranslator._message_to_contents(msg, tc_id_to_name)
            for msg in messages
        ]

        return [content for content in converted if content is not None]

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

        completion_params: "GenerateContentParams" = {
            "model": model,
            "contents": GoogleChatTranslator._messages_to_contents(messages),
            "config": {},
        }

        config: "GenerateContentConfigDict" = {}

        if tools is not None:
            config["tools"] = [GoogleChatTranslator._tool_to_google(t) for t in tools]

        if system_messages := AnthropicChatTranslator._extract_system_messages(
            messages
        ):
            config["system_instruction"] = system_messages

        if params.get("temperature") is not None:
            config["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            config["max_output_tokens"] = params["max_tokens"]

        response_format = params.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_format

        if config:
            completion_params["config"] = config

        return completion_params

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
                for idx, part in enumerate(candidate.content.parts):
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
                                id=f"call_{idx}",
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
                prompt_tokens=raw.usage_metadata.prompt_token_count or 0,
                completion_tokens=raw.usage_metadata.candidates_token_count or 0,
                total_tokens=raw.usage_metadata.total_token_count or 0,
            )

        return CompletionResponse(choices=choices, model=model, usage=usage)
