"""Google Gemini provider using the ``google-genai`` SDK.

Routing prefix: ``google/``

Authentication:
    - Env: ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``
    - Kwargs: ``api_key``

Role mapping:
    - ``system`` -> extracted to ``system_instruction`` config (accepts a list)
    - ``assistant`` -> ``model``
    - ``tool`` -> ``function_response`` part
    - ``user`` -> ``user``

Message constraints:
    - Multiple system messages: supported natively (passed as list)
    - System-only messages: raises ``BadRequestError``
    - No strict alternation required

Tool call format:
    - Tool definitions: converted to ``FunctionDeclaration``
    - Tool results: converted to ``function_response`` parts
    - Tool call IDs: synthetic (``call_0``, ``call_1``, ...) since Gemini
      doesn't provide them

Error mapping (standard ``google.genai.errors``):
    - ``ClientError`` (401/403 or API_KEY_INVALID) -> ``AuthenticationError``
    - ``ClientError`` (429) -> ``RateLimitError``
    - ``ClientError`` (other) -> ``BadRequestError``
    - ``ServerError`` -> ``ServerError``
    - ``APIError`` -> ``LLMError``

Error mapping (Interactions API ``google.genai._interactions``):
    - ``AuthenticationError`` / ``PermissionDeniedError`` -> ``AuthenticationError``
    - ``RateLimitError`` -> ``RateLimitError``
    - ``InternalServerError`` -> ``ServerError``
    - ``APIStatusError`` (other) -> ``BadRequestError``
    - ``APIConnectionError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Structured output (response_format): yes, via ``response_schema``

Provider-specific kwargs:
    - ``safety_settings``: override default safety settings
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import logging
import os
from collections.abc import Sequence
from typing import Any, NoReturn

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
from ..translators.google_chat import GoogleChatTranslator
from ..translators.google_response import GoogleResponseTranslator
from ..types import (
    ChatMessage,
    ChatMessageParam,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    ResponseInputItem,
    ResponseInputItemParam,
    ResponseResult,
    ToolDef,
    ToolDefParam,
)

_CHAT_MESSAGES_TYPE_ADAPTER = TypeAdapter(Sequence[ChatMessage])
_TOOL_DEFS_TYPE_ADAPTER = TypeAdapter(Sequence[ToolDef] | None)
_RESPONSE_INPUT_ITEMS_TYPE_ADAPTER = TypeAdapter(str | Sequence[ResponseInputItem])

logger = logging.getLogger(__name__)

PROVIDER = "google"


KNOWN_EMBEDDING_PARAMS = frozenset({"dimensions"})
KNOWN_RESPONSE_PARAMS = frozenset({"temperature"})
_INSTRUCTION_ROLES = frozenset({"system", "developer"})


def _import_genai() -> Any:
    try:
        from google import genai

        return genai
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "google-genai") from exc


def _import_genai_types() -> Any:
    try:
        from google.genai import types

        return types
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "google-genai") from exc


def _import_genai_errors() -> Any:
    try:
        from google.genai import errors

        return errors
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "google-genai") from exc


def _import_interactions_errors() -> Any:
    try:
        from google.genai import _interactions

        return _interactions
    except (ImportError, AttributeError):
        logger.warning(
            "google.genai._interactions could not be imported; Interactions API error "
            "mapping will be unavailable. This is a private module — verify your "
            "google-genai version if error handling is degraded."
        )
        return None


class GoogleProvider:
    def __init__(
        self,
        api_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        genai = _import_genai()
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = genai.Client(api_key=resolved_key)

    def _map_error(self, e: Exception) -> NoReturn:
        """Map a ``google.genai`` SDK exception to the giskard error hierarchy.

        Handles both the standard ``google.genai.errors`` hierarchy (used by
        Chat Completions / Embeddings) and the separate
        ``google.genai._interactions`` hierarchy (used by Interactions API).
        """
        genai_errors = _import_genai_errors()
        if isinstance(e, genai_errors.ClientError):
            status = getattr(e, "code", 400)
            if status == 429:
                raise RateLimitError(429, str(e), PROVIDER) from e
            if status in (401, 403) or "API_KEY_INVALID" in str(e):
                raise AuthenticationError(status, str(e), PROVIDER) from e
            raise BadRequestError(status, str(e), PROVIDER) from e
        if isinstance(e, genai_errors.ServerError):
            raise ServerError(getattr(e, "code", 500), str(e), PROVIDER) from e
        if isinstance(e, genai_errors.APIError):
            raise LLMError(getattr(e, "code", 500), str(e), PROVIDER) from e

        ix = _import_interactions_errors()
        if ix is not None:
            status = getattr(e, "status_code", 0)
            if isinstance(e, ix.AuthenticationError) or isinstance(
                e, ix.PermissionDeniedError
            ):
                raise AuthenticationError(status or 401, str(e), PROVIDER) from e
            if isinstance(e, ix.RateLimitError):
                raise RateLimitError(status or 429, str(e), PROVIDER) from e
            if isinstance(e, ix.InternalServerError):
                raise ServerError(status or 500, str(e), PROVIDER) from e
            if isinstance(e, ix.APIStatusError):
                if "API_KEY_INVALID" in str(e):
                    raise AuthenticationError(status or 401, str(e), PROVIDER) from e
                raise BadRequestError(status or 400, str(e), PROVIDER) from e
            if isinstance(e, ix.APIConnectionError):
                raise LLMError(0, str(e), PROVIDER) from e
            if isinstance(e, ix.APIError):
                raise LLMError(
                    getattr(e, "status_code", None) or 500, str(e), PROVIDER
                ) from e

        if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
            raise LLMTimeoutError(408, str(e), PROVIDER) from e
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

            kwargs = GoogleChatTranslator.to_google(
                model, messages_models, tools=tools_models, **params
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), PROVIDER) from e

        try:
            raw = await self._client.aio.models.generate_content(**kwargs)
        except Exception as e:  # Broad catch: _map_error checks SDK types first, then applies timeout heuristic, then re-raises.
            self._map_error(e)

        return GoogleChatTranslator.from_google(raw, model, len(messages))

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
                PROVIDER,
                sorted(unknown),
            )

        types = _import_genai_types()

        config_kwargs: dict[str, Any] = {}
        if "dimensions" in params and params["dimensions"] is not None:
            config_kwargs["output_dimensionality"] = params["dimensions"]

        config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        try:
            raw = await self._client.aio.models.embed_content(
                model=model,
                contents=input,
                config=config,
            )
        except Exception as e:  # Broad catch: _map_error checks SDK types first, then applies timeout heuristic, then re-raises.
            self._map_error(e)

        return self._to_embedding_response(raw, model)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(400, "Messages list must not be empty.", PROVIDER)
        has_non_system = any(m.role not in _INSTRUCTION_ROLES for m in messages)
        if not has_non_system:
            raise BadRequestError(
                400, "Messages must contain at least one non-system message.", PROVIDER
            )
        for m in messages:
            if m.role == "tool" and not m.tool_call_id:
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", PROVIDER
                )
            if m.role in _INSTRUCTION_ROLES and not (m.content or "").strip():
                raise BadRequestError(
                    400, "System messages must have non-empty content.", PROVIDER
                )

    # -- helpers ---------------------------------------------------------------

    def _to_embedding_response(self, raw: Any, model: str) -> EmbeddingResponse:
        data: list[EmbeddingData] = []
        if raw.embeddings:
            for i, emb in enumerate(raw.embeddings):
                data.append(EmbeddingData(embedding=list(emb.values), index=i))
        return EmbeddingResponse(data=data, model=model, usage=None)

    # -- Interactions API ------------------------------------------------------

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

            kwargs = GoogleResponseTranslator.to_google(
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
            raw = await self._client.aio.interactions.create(**kwargs)
        except Exception as e:  # Broad catch: _map_error checks SDK types first, then applies timeout heuristic, then re-raises.
            self._map_error(e)

        return GoogleResponseTranslator.from_google(raw, model)
