import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import ClassVar, override

from pydantic import Field, PrivateAttr

from ..limiter.base import RateLimiter, RateLimiterStateRegistry, compute_waited_time


class MaxConcurrentRequestsState:
    """Mutable state for the concurrency limiter."""

    semaphore: asyncio.Semaphore

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)


@RateLimiter.register("max_concurrent")
class MaxConcurrentRequests(RateLimiter):
    """Enforce a maximum number of concurrent in-flight requests."""

    max_concurrent: int = Field(..., ge=1)
    _state_registry: ClassVar[RateLimiterStateRegistry[MaxConcurrentRequestsState]] = (
        RateLimiterStateRegistry[MaxConcurrentRequestsState]()
    )
    _state: MaxConcurrentRequestsState | None = PrivateAttr(default=None)

    async def _get_or_create_state(self) -> MaxConcurrentRequestsState:
        if self._state is not None:
            return self._state

        self._state = await self._state_registry.get_or_create_state(
            (self.id, str(self.max_concurrent)),
            lambda: MaxConcurrentRequestsState(self.max_concurrent),
        )
        return self._state

    @override
    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float, None]:
        state = await self._get_or_create_state()

        start_time = time.monotonic()
        async with state.semaphore:
            end_time = time.monotonic()
            yield compute_waited_time(end_time - start_time)
