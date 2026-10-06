"""OpenAI provider using the ``openai`` SDK.

Routing prefix: ``openai/`` (also the default when no prefix is given)

Authentication:
    - Env: ``OPENAI_API_KEY`` (read by the SDK automatically)
    - Kwargs: ``api_key``, ``base_url``, ``timeout``

Role mapping:
    All canonical roles (system, user, assistant, tool) are passed through
    as-is — OpenAI supports them natively.

Message constraints:
    - Multiple system messages: supported natively
    - System-only messages: raises ``BadRequestError``
    - No strict alternation required

Tool call format:
    Tool definitions and results use the OpenAI format natively.

Error mapping:
    - ``openai.RateLimitError`` -> ``RateLimitError``
    - ``openai.AuthenticationError`` -> ``AuthenticationError``
    - ``openai.BadRequestError`` -> ``BadRequestError``
    - ``openai.APITimeoutError`` -> ``LLMTimeoutError``
    - ``openai.InternalServerError`` -> ``ServerError``
    - ``openai.APIError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Structured output (response_format): yes (passed through to SDK)

Provider-specific kwargs:
    - ``base_url``: custom API endpoint
    - ``timeout``: request timeout in seconds
    - ``http_client``: caller-owned async HTTP client passed to the SDK; not closed by giskard-llm
    - ``default_headers``: extra headers merged into every SDK request

Azure Foundry OpenAI v1:
    Azure Foundry OpenAI v1 endpoints are OpenAI-compatible and should use
    this provider with ``base_url`` set to the Azure ``/openai/v1/`` endpoint.
    The ``model`` argument remains the Azure deployment name for completions,
    Responses API calls, and embeddings.
"""

# pyright: reportAttributeAccessIssue=false

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import TypeAdapter, ValidationError

from ..errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
)
from ..translators.openai_chat import OpenAIChatTranslator
from ..translators.openai_response import OpenAIResponseTranslator
from ..types import (
    ChatMessage,
    ChatMessageParam,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    ResponseInputItem,
    ResponseInputItemParam,
    ResponseResult,
    ToolDef,
    ToolDefParam,
)
from ..utils import compact

if TYPE_CHECKING:
    from httpx import AsyncClient

logger = logging.getLogger(__name__)

PROVIDER = "openai"

KNOWN_EMBEDDING_PARAMS = frozenset({"dimensions"})
_INSTRUCTION_ROLES = frozenset({"system", "developer"})

_CHAT_MESSAGES_TYPE_ADAPTER = TypeAdapter(Sequence[ChatMessage])
_TOOL_DEFS_TYPE_ADAPTER = TypeAdapter(Sequence[ToolDef] | None)
_RESPONSE_INPUT_ITEMS_TYPE_ADAPTER = TypeAdapter(str | Sequence[ResponseInputItem])


def _import_openai() -> Any:
    try:
        import openai

        return openai
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "openai") from exc


class OpenAIProvider:
    _PROVIDER = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: "AsyncClient | None" = None,
        default_headers: Mapping[str, str] | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        openai = _import_openai()
        self._client = openai.AsyncOpenAI(
            **compact(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                http_client=http_client,
                default_headers=default_headers,
            )
        )

    def _map_error(self, e: Exception) -> NoReturn:
        """Map an ``openai.*`` SDK exception to the giskard error hierarchy."""
        openai = _import_openai()
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(429, str(e), self._PROVIDER) from e
        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.BadRequestError):
            raise BadRequestError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.APITimeoutError):
            raise LLMTimeoutError(408, str(e), self._PROVIDER) from e
        if isinstance(e, openai.InternalServerError):
            raise ServerError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.APIError):
            raise LLMError(
                getattr(e, "status_code", None) or 500, str(e), self._PROVIDER
            ) from e
        raise

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessageParam | ChatMessage],
        *,
        tools: Sequence[ToolDefParam | ToolDef] | None = None,
        **params: Any,
    ) -> CompletionResponse:
        try:
            messages_models = _CHAT_MESSAGES_TYPE_ADAPTER.validate_python(messages)
            tools_models = _TOOL_DEFS_TYPE_ADAPTER.validate_python(tools)

            self._validate_messages(messages_models)

            kwargs = OpenAIChatTranslator.to_openai(
                model, messages_models, tools=tools_models, **params
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), PROVIDER) from e

        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except (
            Exception
        ) as e:  # Broad catch: _map_error checks SDK types first, then re-raises.
            self._map_error(e)

        return OpenAIChatTranslator.from_openai(raw)

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        unknown = set(params) - KNOWN_EMBEDDING_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown embedding params: %s",
                self._PROVIDER,
                sorted(unknown),
            )

        kwargs: dict[str, Any] = {"model": model, "input": input}
        if (dimensions := params.get("dimensions")) is not None:
            kwargs["dimensions"] = dimensions
        try:
            raw = await self._client.embeddings.create(**kwargs)
        except (
            Exception
        ) as e:  # Broad catch: _map_error checks SDK types first, then re-raises.
            self._map_error(e)

        return self._to_embedding_response(raw)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(
                400, "Messages list must not be empty.", self._PROVIDER
            )
        has_non_system = any(m.role not in _INSTRUCTION_ROLES for m in messages)
        if not has_non_system:
            raise BadRequestError(
                400,
                "Messages must contain at least one non-system message.",
                self._PROVIDER,
            )
        for m in messages:
            if m.role == "tool" and not m.tool_call_id:
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", self._PROVIDER
                )
            if m.role in _INSTRUCTION_ROLES and not (m.text or "").strip():
                raise BadRequestError(
                    400, "System messages must have non-empty content.", self._PROVIDER
                )

    def _to_embedding_response(self, raw: Any) -> EmbeddingResponse:
        """Convert raw SDK response to EmbeddingResponse."""
        data = [
            EmbeddingData(embedding=item.embedding, index=item.index)
            for item in raw.data
        ]
        usage = None
        if raw.usage:
            usage = EmbeddingUsage(
                prompt_tokens=raw.usage.prompt_tokens,
                total_tokens=raw.usage.total_tokens,
            )
        return EmbeddingResponse(data=data, model=raw.model, usage=usage)

    # -- Responses API ---------------------------------------------------------

    async def respond(
        self,
        model: str,
        input: str | Sequence[ResponseInputItemParam | ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: Sequence[ToolDefParam | ToolDef] | None = None,
        **params: Any,
    ) -> ResponseResult:
        try:
            input_models = _RESPONSE_INPUT_ITEMS_TYPE_ADAPTER.validate_python(input)
            tools_models = _TOOL_DEFS_TYPE_ADAPTER.validate_python(tools)

            kwargs = OpenAIResponseTranslator.to_openai(
                model,
                input_models,
                instructions=instructions,
                previous_id=previous_id,
                tools=tools_models,
                **params,
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), PROVIDER) from e

        try:
            raw = await self._client.responses.create(**kwargs)
        except (
            Exception
        ) as e:  # Broad catch: _map_error checks SDK types first, then re-raises.
            self._map_error(e)

        return OpenAIResponseTranslator.from_openai(raw)
