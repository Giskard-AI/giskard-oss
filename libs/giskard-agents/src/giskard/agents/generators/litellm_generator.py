from typing import cast, override

from litellm import Choices, ModelResponse, acompletion
from litellm import Message as LiteLLMMessage
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ..chat import Message
from .base import BaseGenerator, GenerationParams, Response
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

    @override
    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        params_ = self.params.model_dump(exclude={"tools"})

        if params is not None:
            params_.update(params.model_dump(exclude={"tools"}, exclude_unset=True))

        tools = self.params.tools + (params.tools if params is not None else [])
        if tools:
            params_["tools"] = [t.to_litellm_function() for t in tools]

        response = cast(
            ModelResponse,
            await acompletion(
                messages=[m.to_litellm() for m in messages],
                model=self.model,
                **params_,
            ),
        )

        choice = cast(Choices, response.choices[0])
        return Response(
            message=Message.from_litellm(cast(LiteLLMMessage, choice.message)),
            finish_reason=choice.finish_reason,  # pyright: ignore[reportArgumentType]
        )
