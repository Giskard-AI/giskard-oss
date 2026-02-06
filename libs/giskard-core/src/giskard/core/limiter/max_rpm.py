import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import ClassVar, override

from pydantic import Field, PrivateAttr

from ..limiter.base import RateLimiter, RateLimiterStateRegistry, compute_waited_time


class MaxRequestsPerMinuteState:
    """Mutable state for the per-minute limiter."""

    lock: asyncio.Lock
    next_request_time: float

    def __init__(self):
        self.lock = asyncio.Lock()
        self.next_request_time = time.monotonic()


@RateLimiter.register("max_requests_per_minute")
class MaxRequestsPerMinute(RateLimiter):
    """Enforce a minimum interval between requests."""

    max_requests_per_minute: int = Field(..., ge=1)

    _state_registry: ClassVar[RateLimiterStateRegistry[MaxRequestsPerMinuteState]] = (
        RateLimiterStateRegistry[MaxRequestsPerMinuteState]()
    )
    _state: MaxRequestsPerMinuteState | None = PrivateAttr(default=None)

    @property
    def min_interval(self) -> float:
        return 60.0 / self.max_requests_per_minute

    async def _get_or_create_state(self) -> MaxRequestsPerMinuteState:
        if self._state is not None:
            return self._state

        self._state = await self._state_registry.get_or_create_state(
            (self.id, str(self.max_requests_per_minute)),
            MaxRequestsPerMinuteState,
        )
        return self._state

    @override
    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float, None]:
        state = await self._get_or_create_state()

        async with state.lock:
            current_time = time.monotonic()

            wait_time = state.next_request_time - current_time
            state.next_request_time = (
                max(state.next_request_time, current_time) + self.min_interval
            )

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        yield compute_waited_time(wait_time)
