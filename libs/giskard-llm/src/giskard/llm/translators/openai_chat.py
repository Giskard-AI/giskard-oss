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

from ..types import AssistantMessage
from ..types._serialization import close_object_schemas
from ._unsupported import handle_unsupported_content

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
    }
)
_REASONING_DETAIL_TYPES = frozenset(
    {"reasoning.text", "reasoning.summary", "reasoning.encrypted"}
)


def _keep_reasoning_detail(detail: Any, *, ignore_unsupported_content: bool) -> bool:
    detail_type = detail.get("type") if isinstance(detail, dict) else None
    if detail_type in _REASONING_DETAIL_TYPES:
        return True
    handle_unsupported_content(
        PROVIDER,
        f"reasoning_details:{detail_type}",
        ignore_unsupported_content=ignore_unsupported_content,
    )
    return False


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
        response_format = v.get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = close_object_schemas(response_format.model_json_schema())
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

        # The official Chat Completions API has no reasoning fields on assistant
        # messages (``reasoning`` / ``reasoning_details`` are an OpenRouter / vLLM
        # extension), so strip them rather than send unknown fields.
        messages = [
            m.model_copy(update={"reasoning": None, "reasoning_details": None})
            if isinstance(m, AssistantMessage)
            else m
            for m in messages
        ]

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
        *,
        ignore_unsupported_content: bool = False,
    ) -> "CompletionResponse":
        # OpenAI-compatible servers return ``reasoning`` / ``reasoning_content`` /
        # ``reasoning_details`` as SDK extras, which ``model_dump`` keeps.
        data = raw.model_dump()
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            if details := message.get("reasoning_details"):
                message["reasoning_details"] = [
                    detail
                    for detail in details
                    if _keep_reasoning_detail(
                        detail, ignore_unsupported_content=ignore_unsupported_content
                    )
                ]
        return CompletionResponse.model_validate(data)
