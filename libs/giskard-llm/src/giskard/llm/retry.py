"""Retry eligibility check for LLM provider errors."""

from .errors import LLMTimeoutError, RateLimitError, ServerError

RETRYABLE_ERRORS: frozenset[type[Exception]] = frozenset(
    {LLMTimeoutError, RateLimitError, ServerError}
)


def should_retry(error: Exception) -> bool:
    """Return True if *error* is a transient error worth retrying."""
    return isinstance(error, tuple(RETRYABLE_ERRORS))
