"""Basic rate limiter implementation with RPM and concurrency limits."""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import override

from pydantic import Field, PrivateAttr

from .base import BaseRateLimiter


class _BasicRateLimiterState:
    """Internal state for BasicRateLimiter: semaphore, lock, and next allowed request time."""

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
    """Rate limiter with minimum interval between requests and optional max concurrency.

    Enforces a minimum time between the start of consecutive requests (e.g. RPM limit)
    and optionally limits how many requests can run concurrently.
    """

    min_interval: float = Field(..., ge=0)
    max_concurrent: int | None = Field(default=None, ge=1)

    _state: _BasicRateLimiterState = PrivateAttr()

    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float]:
        """Wait for rate limit, then yield the time waited (or 0.0 if no wait)."""
        start_time = time.monotonic()
        async with self._state.semaphore or nullcontext():
            async with self._state.lock:
                current_time = time.monotonic()
                wait_time = self._state.next_request_time - current_time
                self._state.next_request_time = (
                    max(self._state.next_request_time, current_time) + self.min_interval
                )

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            elapsed_time = time.monotonic() - start_time
            yield 0.0 if elapsed_time <= 1e-3 else elapsed_time

    @override
    def create_initial_state(self) -> _BasicRateLimiterState:
        """Create state with semaphore (if max_concurrent) and lock."""
        return _BasicRateLimiterState(self.max_concurrent)

    @classmethod
    def from_rpm(
        cls, rpm: int, max_concurrent: int | None = None, id: str | None = None
    ) -> "BasicRateLimiter":
        """Create a rate limiter from requests-per-minute (RPM).

        Args:
            rpm: Maximum requests per minute. Must be greater than 0.
            max_concurrent: Maximum concurrent requests allowed, or None for no limit.
            id: Optional unique identifier. Auto-generated if not provided.

        Returns:
            A BasicRateLimiter configured for the given RPM and concurrency.

        Raises:
            ValueError: If rpm is less than or equal to 0.
        """
        if rpm <= 0:
            raise ValueError("RPM must be greater than 0")

        return cls(
            min_interval=60.0 / rpm,
            max_concurrent=max_concurrent,
            id=id or str(uuid.uuid4()),
        )
