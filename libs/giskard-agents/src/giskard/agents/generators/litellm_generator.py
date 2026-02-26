from typing import cast, override

from litellm import Choices, ModelResponse, acompletion
from litellm import Message as LiteLLMMessage
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ..chat import Message
from .base import BaseGenerator, GenerationParams, Response
from .rate_limiting import WithRateLimiter
from .retries import WithRetryPolicy


@BaseGenerator.register("litellm")
class LiteLLMGenerator(WithRateLimiter, WithRetryPolicy, BaseGenerator):
    """A generator for creating chat completion pipelines.

    The MRO places rate limiting inside the retry loop: each retry attempt
    individually acquires the rate limiter. This prevents retry storms from
    bypassing rate limits, at the cost of consuming one rate-limit slot per
    attempt (including failed ones).
    """

    model: str = Field(
        description="The model identifier to use (e.g. 'gemini/gemini-2.0-flash')"
    )

    @override
    def _should_retry(self, err: Exception) -> bool:
        return litellm_should_retry(getattr(err, "status_code", 0))

    @override
    async def _attempt_complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        params_ = self.params.model_dump(exclude={"tools"})

        if params is not None:
            params_.update(params.model_dump(exclude={"tools"}, exclude_unset=True))

        # Now special handling of the tools
        tools = self.params.tools + (params.tools if params is not None else [])
        if tools:
            params_["tools"] = [t.to_litellm_function() for t in tools]

        async with self._throttle():
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
