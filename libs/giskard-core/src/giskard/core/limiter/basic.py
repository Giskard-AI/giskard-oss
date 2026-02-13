import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import override

from pydantic import Field

from ..limiter.base import BaseRateLimiter


class _BasicRateLimiterState:
    semaphore: asyncio.Semaphore | None
    lock: asyncio.Lock
    next_request_time: float

    def __init__(self, max_concurrent: int | None):
        self.semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )
        self.lock = asyncio.Lock()
        self.next_request_time = time.monotonic()


@BaseRateLimiter.register("basic_rate_limiter")
class BasicRateLimiter(BaseRateLimiter):
    min_interval: float = Field(..., ge=0)
    max_concurrent: int | None = Field(default=None, ge=1)

    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float]:
        start_time = time.monotonic()
        async with self.state.semaphore or nullcontext():
            async with self.state.lock:
                current_time = time.monotonic()
                wait_time = self.state.next_request_time - current_time
                self.state.next_request_time = (
                    max(self.state.next_request_time, current_time) + self.min_interval
                )

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            elapsed_time = time.monotonic() - start_time
            yield 0.0 if elapsed_time <= 1e-3 else elapsed_time

    @override
    def create_initial_state(self) -> _BasicRateLimiterState:
        return _BasicRateLimiterState(self.max_concurrent)

    @classmethod
    def from_rpm(
        cls, rpm: int, max_concurrent: int | None = None, id: str | None = None
    ) -> "BasicRateLimiter":
        if rpm <= 0:
            raise ValueError("RPM must be greater than 0")

        return cls(
            min_interval=60.0 / rpm,
            max_concurrent=max_concurrent,
            id=id or str(uuid.uuid4()),
        )
