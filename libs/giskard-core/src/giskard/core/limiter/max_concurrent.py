import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import override

from pydantic import Field

from ..limiter.base import RateLimiterRule


class _MaxConcurrentRequestsState:
    """Mutable state for the concurrency limiter."""

    semaphore: asyncio.Semaphore

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)


@RateLimiterRule.register("max_concurrent")
class MaxConcurrentRequests(RateLimiterRule[_MaxConcurrentRequestsState]):
    """Enforce a maximum number of concurrent in-flight requests."""

    max_concurrent: int = Field(..., ge=1)

    @override
    @asynccontextmanager
    async def throttle(
        self, state: _MaxConcurrentRequestsState
    ) -> AsyncGenerator[None]:
        async with state.semaphore:
            yield

    @override
    def build_initial_state(self) -> _MaxConcurrentRequestsState:
        return _MaxConcurrentRequestsState(self.max_concurrent)
