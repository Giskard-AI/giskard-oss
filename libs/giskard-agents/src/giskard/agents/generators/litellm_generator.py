from typing import cast

from litellm import Choices, ModelResponse, acompletion
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ..chat import Message
from .base import BaseGenerator, GenerationParams, Response
from .mixins import WithRateLimiter, WithRetryPolicy


@BaseGenerator.register("litellm")
class LiteLLMGenerator(WithRateLimiter, WithRetryPolicy, BaseGenerator):
    """A generator for creating chat completion pipelines via LiteLLM."""

    model: str = Field(
        description="The model identifier to use (e.g. 'gemini/gemini-2.0-flash')"
    )

    def _should_retry(self, err: Exception) -> bool:
        return litellm_should_retry(getattr(err, "status_code", 0))

    async def _complete_once(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        params_ = self.params.model_dump(exclude={"tools"})

        if params is not None:
            params_.update(params.model_dump(exclude={"tools"}, exclude_unset=True))

        tools = self.params.tools + (params.tools if params is not None else [])
        if tools:
            params_["tools"] = self.serialize_tools(tools)

        async with self._rate_limiter_context():
            response = cast(
                ModelResponse,
                await acompletion(
                    messages=self.serialize_messages(messages),
                    model=self.model,
                    **params_,
                ),
            )

        choice = cast(Choices, response.choices[0])
        return Response(
            message=self.deserialize_response(choice.message),
            finish_reason=choice.finish_reason,  # pyright: ignore[reportArgumentType]
        )
