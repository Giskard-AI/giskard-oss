from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import AsyncContextManager

import tenacity as t
from pydantic import BaseModel, Field, field_validator

from ..chat import Message
from ..rate_limiter import RateLimiter, get_rate_limiter
from .base import GenerationParams, Response
from .retry import RetryPolicy


class WithRateLimiter(BaseModel):
    """Adds a rate limiter to the generator."""

    rate_limiter: RateLimiter | None = Field(default=None, validate_default=True)

    @field_validator("rate_limiter", mode="before")
    def _validate_rate_limiter(cls, v: RateLimiter | str | None) -> RateLimiter | None:
        if isinstance(v, str):
            v = get_rate_limiter(v)
        return v

    def _rate_limiter_context(self) -> AsyncContextManager:
        if self.rate_limiter is None:
            return nullcontext()

        return self.rate_limiter.throttle()


class WithRetryPolicy(BaseModel, ABC):
    """Adds a retry policy to the generator."""

    retry_policy: RetryPolicy | None = Field(default=RetryPolicy(max_retries=3))

    @abstractmethod
    def _should_retry(self, err: Exception) -> bool: ...

    @abstractmethod
    async def _complete_once(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        """Complete a single request without retry logic.

        This method should be implemented by concrete generators to provide
        the actual completion logic. The retry policy will be applied by
        the _complete method.

        Parameters
        ----------
        messages : list[Message]
            List of messages to send to the model.
        params : GenerationParams | None
            Parameters for the generation.

        Returns
        -------
        Response
            The model's response.
        """
        ...

    def with_retries(
        self,
        max_retries: int,
        *,
        base_delay: float | None = None,
    ) -> "WithRetryPolicy":
        params = {"max_retries": max_retries}

        if base_delay is not None:
            params["base_delay"] = base_delay
        elif self.retry_policy is not None:
            params["base_delay"] = self.retry_policy.base_delay

        return self.model_copy(update={"retry_policy": RetryPolicy(**params)})

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        if self.retry_policy is None:
            return await self._complete_once(messages, params)

        retrier = t.AsyncRetrying(
            stop=t.stop_after_attempt(self.retry_policy.max_retries),
            wait=t.wait_exponential(multiplier=self.retry_policy.base_delay),
            retry=self._tenacity_retry_condition,
            reraise=True,
        )

        return await retrier(self._complete_once, messages, params)

    def _tenacity_retry_condition(self, retry_state: t.RetryCallState) -> bool:
        return self._should_retry(retry_state.outcome.exception())
