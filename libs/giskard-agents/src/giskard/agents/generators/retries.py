from typing import Self

import tenacity as t
from pydantic import BaseModel, Field

from ..chat import Message
from .base import GenerationParams, Response


class RetryPolicy(BaseModel):
    """Configuration for retry behavior.

    Attributes
    ----------
    max_attempts : int
        Maximum number of retry attempts.
    base_delay : float
        Base delay in seconds for exponential backoff.
    max_delay : float | None
        Maximum delay in seconds between retries.
    """

    max_attempts: int = Field(default=3)
    base_delay: float = Field(default=1.0)
    max_delay: float | None = Field(default=None)


class WithRetryPolicy(BaseModel):
    """Adds a retry policy to the generator.

    Note: Subclasses must implement _should_retry and _attempt_complete methods.
    """

    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

    def _should_retry(self, err: Exception) -> bool:
        """Determine if an error should be retried."""
        raise NotImplementedError(
            "WithRetryPolicy mixin requires subclasses to implement _should_retry"
        )

    async def _attempt_complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        """Complete a single request without retry logic.

        This method should be implemented by concrete generators to provide
        the actual completion logic (it replaces the _complete method).

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
        raise NotImplementedError(
            "WithRetryPolicy mixin requires subclasses to implement _attempt_complete"
        )

    def with_retries(
        self,
        max_attempts: int,
        *,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ) -> Self:
        """Create a new generator with updated retry policy.

        Parameters
        ----------
        max_attempts : int
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
        if self.retry_policy is None:
            policy = RetryPolicy()
        else:
            policy = self.retry_policy.model_copy()

        policy.max_attempts = max_attempts

        if base_delay is not None:
            policy.base_delay = base_delay

        if max_delay is not None:
            policy.max_delay = max_delay

        return self.model_copy(update={"retry_policy": policy})

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        """Complete with retry logic applied.

        This method wraps _attempt_complete with the configured retry policy.
        If no retry policy is set, it directly calls _attempt_complete.

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
            return await self._attempt_complete(messages, params)

        wait_kwargs: dict[str, float] = {"multiplier": self.retry_policy.base_delay}
        if self.retry_policy.max_delay is not None:
            wait_kwargs["max"] = self.retry_policy.max_delay

        retrier = t.AsyncRetrying(
            stop=t.stop_after_attempt(self.retry_policy.max_attempts),
            wait=t.wait_exponential(**wait_kwargs),
            retry=self._tenacity_retry_condition,
            before_sleep=self._tenacity_before_sleep,
            reraise=True,
        )

        return await retrier(self._attempt_complete, messages, params)

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
        if retry_state.outcome is None or not retry_state.outcome.failed:
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
