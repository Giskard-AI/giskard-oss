import pytest
from giskard.llm.errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    RateLimitError,
    ServerError,
)
from giskard.llm.retry import should_retry


@pytest.mark.parametrize(
    "error, expected",
    [
        (LLMTimeoutError(408, "timed out", "openai"), True),
        (RateLimitError(429, "rate limited", "openai"), True),
        (ServerError(500, "internal error", "openai"), True),
        (ServerError(502, "bad gateway", "openai"), True),
        (ServerError(503, "unavailable", "openai"), True),
        (AuthenticationError(401, "unauthorized", "openai"), False),
        (BadRequestError(400, "bad request", "openai"), False),
        (LLMError(404, "not found", "openai"), False),
        (ValueError("unrelated error"), False),
    ],
    ids=[
        "timeout",
        "rate-limit",
        "server-500",
        "server-502",
        "server-503",
        "auth-error",
        "bad-request",
        "generic-llm-error",
        "non-llm-error",
    ],
)
def test_should_retry(error: Exception, expected: bool):
    assert should_retry(error) is expected
