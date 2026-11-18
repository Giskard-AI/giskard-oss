import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    CompletionResponse,
    NormalizedReasoningEffort,
    ReasoningContent,
    ToolDef,
)
from giskard.llm.types._base import _BaseModel
from giskard.llm.utils import sanitize_schema_name
from pydantic import BaseModel, SerializationInfo, model_validator

from ..types._serialization import close_object_schemas
from ._unsupported import keep_supported

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
        "reasoning_effort",
    }
)
_REASONING_DETAIL_TYPES = frozenset(
    {"reasoning.text", "reasoning.summary", "reasoning.encrypted"}
)


@AssistantMessage.register_serializer(_PROVIDER)
def serialize_assistant_message(
    message: AssistantMessage, info: SerializationInfo
) -> dict[str, Any]:
    """Serialize an assistant turn without reasoning (ADR-0003).

    Chat Completions has no reasoning field, so ``ReasoningContent`` parts are
    removed from ``content`` (an emptied list becomes no ``content``) and the
    wire-excluded ``reasoning`` field is never sent.
    """
    dump: dict[str, Any] = {"mode": info.mode, "context": info.context}
    data: dict[str, Any] = {"role": message.role}
    content = message.content
    if content is not None and not isinstance(content, str):
        parts = [
            part.model_dump(**dump)
            for part in content
            if not isinstance(part, ReasoningContent)
        ]
        content = parts or None
    if content is not None:
        data["content"] = content
    if message.refusal is not None:
        data["refusal"] = message.refusal
    if message.tool_calls is not None:
        data["tool_calls"] = [
            tool_call.model_dump(**dump) for tool_call in message.tool_calls
        ]
    return data


class OpenAIChatParams(_BaseModel):
    model: str
    messages: Sequence[ChatMessage]
    tools: Sequence[ToolDef] | None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | int | None = None
    metadata: dict[str, str] | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: NormalizedReasoningEffort | None = None

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
        provider: str = PROVIDER,
    ) -> "CompletionResponse":
        # OpenAI-compatible servers return ``reasoning`` / ``reasoning_content`` /
        # ``reasoning_details`` as SDK extras, which ``model_dump`` keeps.
        data = raw.model_dump()
        for choice in data.get("choices") or []:
            if isinstance(message := choice.get("message"), dict):
                _move_reasoning_into_content(
                    message,
                    provider,
                    ignore_unsupported_content=ignore_unsupported_content,
                )
        if isinstance(usage := data.get("usage"), dict):
            details = usage.get("completion_tokens_details")
            if (
                isinstance(details, dict)
                and (reasoning_tokens := details.get("reasoning_tokens")) is not None
            ):
                usage["reasoning_tokens"] = reasoning_tokens
        return CompletionResponse.model_validate(data)


def _move_reasoning_into_content(
    message: dict[str, Any], provider: str, *, ignore_unsupported_content: bool
) -> None:
    """Turn the reasoning extras of a raw message into leading ``reasoning`` parts.

    The raw ``reasoning`` / ``reasoning_content`` / ``reasoning_details`` keys are
    removed, so the deprecated ``AssistantMessage.reasoning`` is derived from the
    parts instead of duplicated.
    """
    details = message.pop("reasoning_details", None)
    # vLLM names it ``reasoning``, DeepSeek ``reasoning_content``.
    text = message.pop("reasoning", None)
    reasoning_content = message.pop("reasoning_content", None)
    if text is None:
        text = reasoning_content

    parts: list[dict[str, Any]] = []
    if details:
        kept = keep_supported(
            details,
            _REASONING_DETAIL_TYPES,
            provider,
            ignore_unsupported_content=ignore_unsupported_content,
            label_prefix="reasoning_details:",
        )
        parts = [_detail_to_part(detail, provider) for detail in kept]
    elif text:
        parts = [_reasoning_part(provider, content=[_reasoning_text(text)])]
    if not parts:
        return

    content = message.get("content")
    if content is None:
        message["content"] = parts
    elif isinstance(content, str):
        message["content"] = [*parts, {"type": "text", "text": content}]
    else:
        message["content"] = [*parts, *content]


def _reasoning_text(text: str) -> dict[str, Any]:
    return {"type": "reasoning_text", "text": text}


def _reasoning_part(provider: str, **fields: Any) -> dict[str, Any]:
    return {"type": "reasoning", "provider": provider, **fields}


def _detail_to_part(detail: dict[str, Any], provider: str) -> dict[str, Any]:
    """Map one OpenRouter ``reasoning_details`` item to one reasoning part.

    ``format`` and ``index`` are ignored: the part is tagged with the route instead.
    """
    item_id = detail.get("id")
    match detail["type"]:
        case "reasoning.text":
            return _reasoning_part(
                provider,
                id=item_id,
                content=[_reasoning_text(detail.get("text") or "")],
                signature=detail.get("signature"),
            )
        case "reasoning.summary":
            return _reasoning_part(
                provider,
                id=item_id,
                summary=[{"type": "summary_text", "text": detail.get("summary") or ""}],
            )
        case _:  # "reasoning.encrypted"
            return _reasoning_part(
                provider, id=item_id, encrypted_content=detail.get("data")
            )
