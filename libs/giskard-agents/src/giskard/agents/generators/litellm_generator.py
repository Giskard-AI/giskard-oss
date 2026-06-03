"""Optional LiteLLM-backed generator.

Install the optional dependency with::

    pip install giskard-agents[litellm]

Importing this module is safe without litellm; instantiation raises ImportError.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, override

from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field

from ..tools import Tool
from ._types import GenerationParams
from .base import BaseGenerator
from .middleware import CompletionMiddleware, RetryMiddleware, RetryPolicy

if TYPE_CHECKING:
    from litellm import ModelResponse
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )


def _import_litellm() -> Any:
    try:
        import litellm

        return litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLMGenerator requires the optional 'litellm' dependency. "
            "Install it with: pip install giskard-agents[litellm]"
        ) from exc


@CompletionMiddleware.register("litellm_retry")
class LiteLLMRetryMiddleware(RetryMiddleware):
    """Retry middleware using LiteLLM's built-in retry-eligibility check."""

    @override
    def _should_retry(self, err: Exception) -> bool:
        litellm = _import_litellm()
        return litellm._should_retry(getattr(err, "status_code", 0))


@BaseGenerator.register("litellm")
class LiteLLMGenerator(BaseGenerator):
    """A generator for creating chat completion pipelines using LiteLLM."""

    model: str = Field(
        description="The model identifier to use (e.g. 'gemini/gemini-3.5-flash')"
    )
    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

    def model_post_init(self, __context: Any) -> None:
        """Fail fast if litellm is not installed."""
        super().model_post_init(__context)
        _import_litellm()

    @override
    def _create_retry_middleware(self) -> LiteLLMRetryMiddleware | None:
        if self.retry_policy is None:
            return None
        return LiteLLMRetryMiddleware(retry_policy=self.retry_policy)

    def _serialize_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert ``Tool`` objects to the OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in tools
        ]

    def _serialize_messages(
        self, messages: Sequence[ChatMessage]
    ) -> "list[ChatCompletionMessageParam]":
        """Convert ``Message`` objects to OpenaAI's dict format (litellm expects)."""
        return [
            cast(
                "ChatCompletionMessageParam",
                cast(object, m.model_dump(context={"provider": "openai/chat"})),
            )
            for m in messages
        ]

    def _deserialize_response(self, raw: Any) -> AssistantMessage:
        """Convert a LiteLLM response object into an internal ``Message``."""
        data = raw if isinstance(raw, dict) else raw.model_dump()
        return AssistantMessage.model_validate(data)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        litellm = _import_litellm()
        wire_messages = self._serialize_messages(messages)
        wire_params = params.model_dump(exclude={"tools"}, exclude_unset=True)
        wire_tools = self._serialize_tools(params.tools) if params.tools else []
        if wire_tools:
            wire_params["tools"] = wire_tools
        if metadata:
            wire_params["metadata"] = metadata

        raw = cast(
            "ModelResponse",
            await litellm.acompletion(
                messages=wire_messages, model=self.model, **wire_params
            ),
        )

        return CompletionResponse(
            choices=[
                Choice(
                    message=self._deserialize_response(choice.message),
                    finish_reason=choice.finish_reason,
                )
                for choice in raw.choices
            ]
        )
