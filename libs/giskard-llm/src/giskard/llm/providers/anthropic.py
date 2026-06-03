"""Anthropic provider using the ``anthropic`` SDK.

Routing prefix: ``anthropic/``

Authentication:
    - Env: ``ANTHROPIC_API_KEY`` (read by the SDK automatically)
    - Kwargs: ``api_key``, ``base_url``, ``timeout``

Role mapping:
    - ``system`` / ``developer`` -> extracted to top-level ``system`` param (text blocks)
    - ``user`` -> ``user``
    - ``assistant`` -> ``assistant``
    - ``tool`` -> wrapped as ``user`` with ``tool_result`` content block

Message constraints:
    - Multiple system messages: raises ``BadRequestError`` by default.
      Configure with ``merge_system=True`` to concatenate them.
    - Consecutive same-role messages: raises ``BadRequestError``
      (strict alternation required by the Anthropic API).
    - System- or developer-only messages: raises ``BadRequestError``

Tool call format:
    - Tool definitions: converted to Anthropic ``{name, description, input_schema}``
    - Tool results: converted to ``tool_result`` content blocks in ``user`` messages
    - Tool call IDs: preserved from Anthropic's ``tool_use`` blocks

Error mapping:
    - ``anthropic.RateLimitError`` -> ``RateLimitError``
    - ``anthropic.AuthenticationError`` -> ``AuthenticationError``
    - ``anthropic.BadRequestError`` -> ``BadRequestError``
    - ``anthropic.APITimeoutError`` -> ``LLMTimeoutError``
    - ``anthropic.InternalServerError`` -> ``ServerError``
    - ``anthropic.APIStatusError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: no (provider does not implement ``EmbeddingProvider``)
    - Structured output (response_format): yes, via native ``output_config`` (json_schema)

Provider-specific kwargs (configure-time):
    - ``merge_system``: if True, concatenate multiple system messages instead of raising
    - ``base_url``: custom API endpoint
    - ``timeout``: request timeout in seconds
    - ``http_client``: caller-owned async HTTP client passed to the SDK; not closed by giskard-llm
    - ``default_headers``: extra headers merged into every SDK request
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

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
from ..translators.anthropic import AnthropicChatTranslator
from ..types import (
    ChatMessage,
    ChatMessageParam,
    CompletionResponse,
    ToolDef,
    ToolDefParam,
)
from ..utils import compact

if TYPE_CHECKING:
    from httpx import AsyncClient

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"

# Roles that never appear in ``messages`` on the wire (folded into top-level ``system``).
_ANTHROPIC_INSTRUCTION_ROLES = frozenset({"system", "developer"})


_CHAT_MESSAGES_TYPE_ADAPTER = TypeAdapter(Sequence[ChatMessage])
_TOOL_DEFS_TYPE_ADAPTER = TypeAdapter(Sequence[ToolDef] | None)
# -- Private wire-format TypedDicts -------------------------------------------


def _import_anthropic() -> Any:
    try:
        import anthropic

        return anthropic
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "anthropic") from exc


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        merge_system: bool = False,
        http_client: "AsyncClient | None" = None,
        default_headers: Mapping[str, str] | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        anthropic = _import_anthropic()
        self._merge_system = merge_system
        self._client = anthropic.AsyncAnthropic(
            **compact(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                http_client=http_client,
                default_headers=default_headers,
            )
        )

    def _map_error(self, e: Exception) -> NoReturn:
        """Map an ``anthropic.*`` SDK exception to the giskard error hierarchy."""
        anthropic = _import_anthropic()
        if isinstance(e, anthropic.RateLimitError):
            raise RateLimitError(429, str(e), PROVIDER) from e
        if isinstance(e, anthropic.AuthenticationError):
            raise AuthenticationError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.BadRequestError):
            raise BadRequestError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APITimeoutError):
            raise LLMTimeoutError(408, str(e), PROVIDER) from e
        if isinstance(e, anthropic.InternalServerError):
            raise ServerError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APIStatusError):
            raise LLMError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APIError):
            raise LLMError(
                getattr(e, "status_code", None) or 500, str(e), PROVIDER
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

            kwargs = AnthropicChatTranslator.to_anthropic(
                model, messages_models, tools=tools_models, **params
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), PROVIDER) from e

        try:
            raw = await self._client.messages.create(**kwargs)
        except (
            Exception
        ) as e:  # Broad catch: _map_error checks SDK types first, then re-raises.
            self._map_error(e)

        return AnthropicChatTranslator.from_anthropic(raw)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(400, "Messages list must not be empty.", PROVIDER)

        system_count = sum(
            1 for m in messages if m.role in _ANTHROPIC_INSTRUCTION_ROLES
        )
        has_conversation_message = any(
            m.role not in _ANTHROPIC_INSTRUCTION_ROLES for m in messages
        )
        if not has_conversation_message:
            raise BadRequestError(
                400,
                "Messages must contain at least one non-system message.",
                PROVIDER,
            )

        if system_count > 1 and not self._merge_system:
            raise BadRequestError(
                400,
                "Anthropic does not support multiple system messages. "
                "Configure with merge_system=True to concatenate them.",
                PROVIDER,
            )

        for_alternation = [
            m for m in messages if m.role not in _ANTHROPIC_INSTRUCTION_ROLES
        ]
        for i in range(1, len(for_alternation)):
            prev_role = for_alternation[i - 1].role
            curr_role = for_alternation[i].role
            # Skip: consecutive tool messages are valid (they merge into
            # a single user message with multiple tool_result blocks).
            if prev_role == "tool" or curr_role == "tool":
                continue
            if prev_role == curr_role:
                raise BadRequestError(
                    400,
                    f"Anthropic requires alternating user/assistant messages, "
                    f"but found consecutive '{curr_role}' messages.",
                    PROVIDER,
                )

        for m in messages:
            if m.role == "tool" and not m.tool_call_id:
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", PROVIDER
                )
            if m.role in _ANTHROPIC_INSTRUCTION_ROLES and not (m.content or "").strip():
                raise BadRequestError(
                    400,
                    "System and developer messages must have non-empty content.",
                    PROVIDER,
                )
