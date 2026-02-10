import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import override

from pydantic import Field

from ..limiter.base import RateLimiterRule, compute_waited_time


class _MinIntervalState:
    """Mutable state for the interval limiter."""

    lock: asyncio.Lock = asyncio.Lock()
    next_request_time: float = time.monotonic()


@RateLimiterRule.register("min_interval")
class MinInterval(RateLimiterRule[_MinIntervalState]):
    """Enforce a minimum interval between requests."""

    min_interval: float = Field(..., ge=0)

    @override
    @asynccontextmanager
    async def throttle(self, state: _MinIntervalState) -> AsyncGenerator[float, None]:
        async with state.lock:
            current_time = time.monotonic()

            wait_time = state.next_request_time - current_time
            state.next_request_time = (
                max(state.next_request_time, current_time) + self.min_interval
            )

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        yield compute_waited_time(wait_time)

    @override
    def build_initial_state(self) -> _MinIntervalState:
        return _MinIntervalState()
