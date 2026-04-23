from typing import Any, cast, override

from giskard.llm import CompletionResponse, acompletion, should_retry
from giskard.llm.types import ChatMessageParam
from pydantic import Field

from ..chat import Message
from ..tools import Tool
from ._types import FinishReason, GenerationParams, Response
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
        description="The model identifier to use (e.g. 'google/gemini-2.0-flash')"
    )
    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

    @override
    def _create_retry_middleware(self) -> GiskardLLMRetryMiddleware | None:
        if self.retry_policy is None:
            return None
        return GiskardLLMRetryMiddleware(retry_policy=self.retry_policy)

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

    def _serialize_messages(self, messages: list[Message]) -> list[ChatMessageParam]:
        """Convert ``Message`` objects to the wire dict format."""
        return cast(
            list[ChatMessageParam],
            [
                m.model_dump(include={"role", "content", "tool_calls", "tool_call_id"})
                for m in messages
            ],
        )

    def _deserialize_response(self, raw: Any) -> Message:
        """Convert a response message object into an internal ``Message``."""
        data = raw if isinstance(raw, dict) else raw.model_dump()
        return Message.model_validate(data)

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        wire_messages = self._serialize_messages(messages)
        wire_params = params.model_dump(exclude={"tools"})
        wire_tools = self._serialize_tools(params.tools) if params.tools else []
        if wire_tools:
            wire_params["tools"] = wire_tools
        if metadata:
            wire_params["metadata"] = metadata

        raw: CompletionResponse = await acompletion(
            messages=wire_messages, model=self.model, **wire_params
        )

        choice = raw.choices[0]
        message = self._deserialize_response(choice.message)
        response_metadata = raw.model_dump(exclude={"choices"})
        return Response(
            message=message,
            finish_reason=cast(FinishReason, choice.finish_reason),
            metadata=response_metadata,
        )
