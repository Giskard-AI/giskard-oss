from typing import Any

from .base import RateLimiter
from .max_concurrent import MaxConcurrentRequests
from .max_rpm import MaxRequestsPerMinute


def rpm(requests: int, id: str | None = None) -> MaxRequestsPerMinute:
    """
    Creates a limiter that enforces a maximum number of requests per minute.

    If an ID is provided, multiple instances with the same ID will share
    the same internal lock/state.
    """
    kwargs: dict[str, Any] = {"max_requests_per_minute": requests}
    if id is not None:
        kwargs["id"] = id

    return MaxRequestsPerMinute(**kwargs)


def max_concurrent(limit: int, id: str | None = None) -> MaxConcurrentRequests:
    """
    Creates a limiter that enforces a maximum number of concurrent requests.

    If an ID is provided, multiple instances with the same ID will share
    the same internal lock/state.
    """
    kwargs: dict[str, Any] = {"max_concurrent": limit}
    if id is not None:
        kwargs["id"] = id
    return MaxConcurrentRequests(**kwargs)


__all__ = [
    "rpm",
    "max_concurrent",
    "MaxRequestsPerMinute",
    "MaxConcurrentRequests",
    "RateLimiter",
]
