"""Protocol definitions for LLM providers.

Providers implement whichever protocols they support. The router checks
capability at dispatch time via ``isinstance``.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..types import (
    ChatMessage,
    ChatMessageParam,
    CompletionResponse,
    EmbeddingResponse,
    ResponseInputItem,
    ResponseInputItemParam,
    ResponseResult,
    ToolDef,
    ToolDefParam,
)


@runtime_checkable
class CompletionProvider(Protocol):
    """Provider capable of chat completions."""

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessageParam | ChatMessage],
        *,
        tools: Sequence[ToolDefParam | ToolDef] | None = None,
        **params: Any,
    ) -> CompletionResponse:
        """Send a chat completion request. Raises LLMError on provider failures."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider capable of text embeddings."""

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        """Compute embeddings for the given texts. Raises LLMError on provider failures."""
        ...


@runtime_checkable
class ResponseProvider(Protocol):
    """Provider capable of stateful responses (OpenAI Responses / Gemini Interactions)."""

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
        """Send a stateful response request. ``input`` is a string or structured items for multi-turn."""
        ...
