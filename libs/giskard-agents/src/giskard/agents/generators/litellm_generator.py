from typing import Any, cast, override

from litellm import Choices, ModelResponse, acompletion
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ._types import FinishReason
from .base import BaseGenerator
from .middleware import CompletionMiddleware, RetryMiddleware


@CompletionMiddleware.register("litellm_retry")
class LiteLLMRetryMiddleware(RetryMiddleware):
    """Retry middleware using LiteLLM's built-in retry-eligibility check."""

    @override
    def _should_retry(self, err: Exception) -> bool:
        return litellm_should_retry(getattr(err, "status_code", 0))


@BaseGenerator.register("litellm")
class LiteLLMGenerator(BaseGenerator):
    """A generator for creating chat completion pipelines using LiteLLM."""

    model: str = Field(
        description="The model identifier to use (e.g. 'gemini/gemini-2.0-flash')"
    )
    middleware: list[CompletionMiddleware] = Field(
        default_factory=lambda: [LiteLLMRetryMiddleware()]
    )

    @override
    async def _call_model(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> tuple[Any, FinishReason]:
        if tools:
            params = {**params, "tools": tools}

        response = cast(
            ModelResponse,
            await acompletion(messages=messages, model=self.model, **params),
        )

        choice = cast(Choices, response.choices[0])
        return choice.message, choice.finish_reason  # pyright: ignore[reportReturnType]
