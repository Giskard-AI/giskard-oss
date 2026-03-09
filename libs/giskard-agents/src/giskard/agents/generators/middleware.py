"""Completion middleware pipeline for generators.

Middleware intercepts the completion call, wrapping the inner function with
cross-cutting concerns (retries, rate limiting, logging, ...).  Each middleware
is a Pydantic model in a discriminated union, so the full pipeline serializes
and round-trips via ``model_dump_json`` / ``model_validate_json``.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import tenacity as t
from giskard.core import BaseRateLimiter, Discriminated, discriminated_base
from pydantic import BaseModel, Field

from ..chat import Message
from ._types import GenerationParams, Response

type NextFn = Callable[[list[Message], GenerationParams | None], Awaitable[Response]]


@discriminated_base
class CompletionMiddleware(Discriminated, ABC):
    """Base class for completion middleware.

    Subclasses wrap the ``next_fn`` with cross-cutting logic (retries,
    throttling, caching, logging, …) and must be registered with
    ``@CompletionMiddleware.register("name")``.
    """

    @abstractmethod
    async def call(
        self,
        messages: list[Message],
        params: GenerationParams | None,
        next_fn: NextFn,
    ) -> Response:
        """Invoke the middleware, calling *next_fn* to continue the chain."""


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3)
    base_delay: float = Field(default=1.0)
    max_delay: float | None = Field(default=None)


@CompletionMiddleware.register("retry")
class RetryMiddleware(CompletionMiddleware):
    """Retries failed completions with exponential back-off (tenacity).

    Override ``_should_retry`` in subclasses for provider-specific logic.
    """

    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    def _should_retry(self, err: Exception) -> bool:  # noqa: ARG002
        """Return ``True`` if *err* warrants a retry.  Default: retry all."""
        return True

    async def call(
        self,
        messages: list[Message],
        params: GenerationParams | None,
        next_fn: NextFn,
    ) -> Response:
        policy = self.retry_policy
        wait_kwargs: dict[str, float] = {"multiplier": policy.base_delay}
        if policy.max_delay is not None:
            wait_kwargs["max"] = policy.max_delay

        retrier = t.AsyncRetrying(
            stop=t.stop_after_attempt(policy.max_attempts),
            wait=t.wait_exponential(**wait_kwargs),
            retry=self._tenacity_retry_condition,
            reraise=True,
        )

        return await retrier(next_fn, messages, params)

    def _tenacity_retry_condition(self, retry_state: t.RetryCallState) -> bool:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return False
        return self._should_retry(retry_state.outcome.exception())  # pyright: ignore[reportArgumentType]


@CompletionMiddleware.register("rate_limiter")
class RateLimiterMiddleware(CompletionMiddleware):
    """Throttles completions through a :class:`BaseRateLimiter`."""

    rate_limiter: BaseRateLimiter

    async def call(
        self,
        messages: list[Message],
        params: GenerationParams | None,
        next_fn: NextFn,
    ) -> Response:
        async with self.rate_limiter.throttle():
            return await next_fn(messages, params)
