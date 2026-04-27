import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ChatMessage,
    CompletionResponse,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
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


class OpenAIChatParams(_BaseModel):
    model: str
    messages: Sequence[ChatMessage]
    tools: Sequence[ToolDef] | None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | int | None = None
    metadata: dict[str, str] | None = None
    response_format: dict[str, object] | None = None

    @field_validator("response_format")
    @staticmethod
    def _coerce_response_format(
        v: type | dict[str, object] | None,
    ) -> type | dict[str, object] | None:
        if not isinstance(v, type) or not issubclass(v, BaseModel):
            return v

        schema = v.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "json_schema",
            "json_schema": {
                "name": v.__name__,
                "strict": True,
                "schema": schema,
            },
        }


class OpenAIChatTranslator:
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

        chat_params = OpenAIChatParams.model_validate(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                **params,
            }
        )

        return cast(
            "CompletionCreateParamsWithTimeout",
            cast(object, chat_params.model_dump(context={"provider": "openai/chat"})),
        )

    @staticmethod
    def from_openai(
        raw: "ChatCompletion",
    ) -> "CompletionResponse":
        return CompletionResponse.model_validate(raw.model_dump())
