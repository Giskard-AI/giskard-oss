import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    ChatMessage,
    CompletionResponse,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from giskard.llm.utils import sanitize_schema_name
from pydantic import BaseModel, model_validator

from ..structured_output import normalize_pydantic_json_schema, pop_schema_mutation

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
_PROVIDER = "openai/chat"
KNOWN_COMPLETION_PARAMS = frozenset(
    {
        "temperature",
        "max_tokens",
        "timeout",
        "tools",
        "response_format",
        "metadata",
        "schema_mutation",
    }
)


class OpenAIChatParams(_BaseModel):
    model: str
    messages: Sequence[ChatMessage]
    tools: Sequence[ToolDef] | None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | int | None = None
    metadata: dict[str, str] | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_response_format_and_strip_internal(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = v.copy()
        mode = pop_schema_mutation(v)
        response_format = v.get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = normalize_pydantic_json_schema(
                response_format,
                profile="openai",
                provider=PROVIDER,
                mode=mode,
            )
            v["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": sanitize_schema_name(response_format.__name__),
                    "schema": schema,
                },
            }
        return v


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
            cast(object, chat_params.model_dump(context={"provider": _PROVIDER})),
        )

    @staticmethod
    def from_openai(
        raw: "ChatCompletion",
    ) -> "CompletionResponse":
        return CompletionResponse.model_validate(raw.model_dump())
