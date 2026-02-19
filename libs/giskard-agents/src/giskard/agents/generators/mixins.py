from contextlib import asynccontextmanager
from typing import Any, Self

import tenacity as t
from giskard.core import BaseRateLimiter
from pydantic import BaseModel, Field

from ..chat import Message
from .base import GenerationParams, Response
from .retry import MAX_WAIT_SECONDS, RetryPolicy


class WithRateLimiter(BaseModel):
    """Adds a rate limiter to the generator."""

    rate_limiter: BaseRateLimiter | None = Field(default=None)

    def with_rate_limiter(self, rate_limiter: BaseRateLimiter | str | None) -> Self:
        if isinstance(rate_limiter, str):
            rate_limiter = BaseRateLimiter.from_id(rate_limiter)

        return self.model_copy(update={"rate_limiter": rate_limiter})

    @asynccontextmanager
    async def _throttle(self):
        if self.rate_limiter is None:
            yield 0.0
            return

        async with self.rate_limiter.throttle() as waited:
            yield waited


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
        max_delay: float | None = None,
    ) -> "WithRetryPolicy":
        """Create a new generator with updated retry policy.

        Parameters
        ----------
        max_retries : int
            Maximum number of retry attempts.
        base_delay : float | None
            Base delay in seconds for exponential backoff. If None, preserves existing value.
        max_delay : float | None
            Maximum delay in seconds between retries. If None, preserves existing value.

        Returns
        -------
        WithRetryPolicy
            A new generator with the updated retry policy.
        """
        patch: dict[str, Any] = {"max_retries": max_retries}

        if base_delay is not None:
            patch["base_delay"] = base_delay
        elif self.retry_policy is not None:
            patch["base_delay"] = self.retry_policy.base_delay

        if max_delay is not None:
            patch["max_delay"] = max_delay
        elif self.retry_policy is not None:
            patch["max_delay"] = self.retry_policy.max_delay

        return self.model_copy(update={"retry_policy": RetryPolicy(**patch)})

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        """Complete with retry logic applied.

        This method wraps _complete_once with the configured retry policy.
        If no retry policy is set, it directly calls _complete_once.

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
        if self.retry_policy is None:
            return await self._complete_once(messages, params)

        retrier = t.AsyncRetrying(
            stop=t.stop_after_attempt(self.retry_policy.max_retries),
            wait=t.wait_exponential(
                multiplier=self.retry_policy.base_delay,
                max=(
                    self.retry_policy.max_delay
                    if self.retry_policy.max_delay is not None
                    else MAX_WAIT_SECONDS
                ),
            ),
            retry=self._tenacity_retry_condition,
            before_sleep=self._tenacity_before_sleep,
            reraise=True,
        )

        return await retrier(self._complete_once, messages, params)

    def _tenacity_retry_condition(self, retry_state: t.RetryCallState) -> bool:
        """Determine if a retry should be attempted based on the outcome.

        Parameters
        ----------
        retry_state : t.RetryCallState
            The current state of the retry attempt.

        Returns
        -------
        bool
            True if the error should trigger a retry, False otherwise.
        """
        if retry_state.outcome is None:
            return False

        return self._should_retry(retry_state.outcome.exception())  # pyright: ignore[reportArgumentType]

    def _tenacity_before_sleep(self, retry_state: t.RetryCallState) -> None:
        """Hook called before sleeping between retry attempts.

        This method can be overridden by subclasses to perform custom actions
        before each retry sleep (e.g., logging, metrics collection).

        Parameters
        ----------
        retry_state : t.RetryCallState
            The current state of the retry attempt.
        """
        pass
