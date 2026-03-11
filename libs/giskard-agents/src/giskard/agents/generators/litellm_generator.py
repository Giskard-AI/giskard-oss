from typing import Any, cast, override

from litellm import Choices, ModelResponse, acompletion
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ..chat import Message
from ..tools import Tool
from ._types import FinishReason, GenerationParams
from .base import BaseGenerator
from .middleware import CompletionMiddleware, RetryMiddleware, RetryPolicy


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
    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

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

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert ``Message`` objects to LiteLLM's dict format."""
        return [
            m.model_dump(include={"role", "content", "tool_calls", "tool_call_id"})
            for m in messages
        ]

    def _deserialize_response(self, raw: Any) -> Message:
        """Convert a LiteLLM response object into an internal ``Message``."""
        data = raw if isinstance(raw, dict) else raw.model_dump()
        return Message.model_validate(data)

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
    ) -> tuple[Message, FinishReason]:
        wire_messages = self._serialize_messages(messages)
        wire_params = params.model_dump(exclude={"tools"})
        wire_tools = self._serialize_tools(params.tools) if params.tools else []
        if wire_tools:
            wire_params["tools"] = wire_tools

        response = cast(
            ModelResponse,
            await acompletion(messages=wire_messages, model=self.model, **wire_params),
        )

        choice = cast(Choices, response.choices[0])
        message = self._deserialize_response(choice.message)
        return message, choice.finish_reason  # pyright: ignore[reportReturnType]
