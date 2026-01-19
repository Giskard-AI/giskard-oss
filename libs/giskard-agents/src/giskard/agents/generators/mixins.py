from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any, cast

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
        # Supported singleton semantics are implemented at the container level:
        # - If a string is provided, it must already exist in the registry.
        # - If a dict is provided (e.g. from JSON deserialization), we reuse an
        #   already-registered instance when possible, otherwise we let Pydantic
        #   create a new RateLimiter from the dict.
        if v is None or isinstance(v, RateLimiter):
            return v

        if isinstance(v, str):
            return get_rate_limiter(v)

        if isinstance(v, dict):
            rate_limiter_id = v.get("rate_limiter_id")
            if rate_limiter_id:
                try:
                    return get_rate_limiter(rate_limiter_id)
                except ValueError:
                    return v  # let Pydantic create & register a new instance
            return v

        return v

    def _rate_limiter_context(
        self,
    ) -> AbstractAsyncContextManager[RateLimiter | None, None]:
        if self.rate_limiter is None:
            return nullcontext(None)

        return cast(
            AbstractAsyncContextManager[RateLimiter | None, None],
            self.rate_limiter.throttle(),
        )


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
