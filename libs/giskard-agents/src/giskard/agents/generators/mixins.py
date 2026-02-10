from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Self, overload

import tenacity as t
from giskard.core import RateLimiter
from giskard.core.limiter.base import NO_THROTTLE
from pydantic import BaseModel, Field

from ..chat import Message
from .base import GenerationParams, Response
from .retry import RetryPolicy


class WithRateLimiters(BaseModel):
    """Adds a rate limiter to the generator."""

    rate_limiter: list[RateLimiter] | RateLimiter | None = Field(default=None)

    @overload
    def with_rate_limiter(self, rate_limiter: None, /) -> Self: ...

    @overload
    def with_rate_limiter(self, *rate_limiters: str | RateLimiter) -> Self: ...

    def with_rate_limiter(self, *rate_limiters: str | RateLimiter | None) -> Self:
        limiters = [
            RateLimiter.from_id(rate_limiter)
            if isinstance(rate_limiter, str)
            else rate_limiter
            for rate_limiter in rate_limiters
            if rate_limiter is not None
        ]

        return self.model_copy(
            update={"rate_limiter": None if len(limiters) == 0 else limiters}
        )

    @asynccontextmanager
    async def _throttle(self):
        throttled_events = NO_THROTTLE

        limiters = self.rate_limiter
        if not limiters:
            yield throttled_events
            return

        if isinstance(limiters, RateLimiter):
            limiters = [limiters]

        async with AsyncExitStack() as stack:
            for rate_limiter in limiters:
                throttled_events += await stack.enter_async_context(
                    rate_limiter.throttle()
                )

            yield throttled_events


class WithRetryPolicy(BaseModel):
    """Adds a retry policy to the generator.

    Note: Subclasses must implement _should_retry and _complete_once methods.
    These are enforced when mixed with BaseGenerator (which inherits from ABC).
    """

    retry_policy: RetryPolicy | None = Field(default=RetryPolicy(max_retries=3))

    def _should_retry(self, err: Exception) -> bool:
        """Determine if an error should be retried.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _should_retry")

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
        raise NotImplementedError("Subclasses must implement _complete_once")

    def with_retries(
        self,
        max_retries: int,
        *,
        base_delay: float | None = None,
    ) -> "WithRetryPolicy":
        params: dict[str, Any] = {"max_retries": max_retries}

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
        if retry_state.outcome is None:
            return False

        return self._should_retry(retry_state.outcome.exception())  # pyright: ignore[reportArgumentType]
