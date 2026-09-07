from collections.abc import Sequence
from typing import Any, override

from giskard.llm import CompletionResponse, acompletion, should_retry
from giskard.llm.types import ChatMessage, ToolDefParam
from pydantic import Field

from ..tools import Tool
from ._types import GenerationParams
from .base import BaseGenerator
from .middleware import CompletionMiddleware, RetryMiddleware, RetryPolicy


@CompletionMiddleware.register("giskard_llm_retry")
class GiskardLLMRetryMiddleware(RetryMiddleware):
    """Retry middleware that checks error types for retry eligibility."""

    @override
    def _should_retry(self, err: Exception) -> bool:
        return should_retry(err)


@BaseGenerator.register("giskard_llm")
class GiskardLLMGenerator(BaseGenerator):
    """A generator for creating chat completion pipelines."""

    model: str = Field(
        description="The model identifier to use (e.g. 'google/gemini-3.5-flash')"
    )
    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

    @override
    def _create_retry_middleware(self) -> GiskardLLMRetryMiddleware | None:
        if self.retry_policy is None:
            return None
        return GiskardLLMRetryMiddleware(retry_policy=self.retry_policy)

    def _serialize_tools(self, tools: list[Tool]) -> list[ToolDefParam]:
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

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        wire_params = params.model_dump(exclude={"tools"}, exclude_unset=True)
        wire_tools = self._serialize_tools(params.tools) if params.tools else []
        if wire_tools:
            wire_params["tools"] = wire_tools
        if metadata:
            wire_params["metadata"] = metadata

        return await acompletion(messages=messages, model=self.model, **wire_params)
