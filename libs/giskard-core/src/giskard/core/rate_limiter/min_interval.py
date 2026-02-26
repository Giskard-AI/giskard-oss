"""Basic rate limiter implementation with RPM and concurrency limits."""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import Self, override

from pydantic import Field, PrivateAttr

from .base import BaseRateLimiter


class _MinIntervalRateLimiterState:
    """Internal state for MinIntervalRateLimiter: semaphore, lock, and next allowed request time."""

    semaphore: asyncio.Semaphore | None
    lock: asyncio.Lock
    next_request_time: float

    def __init__(self, max_concurrent: int | None):
        self.semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )
        self.lock = asyncio.Lock()
        self.next_request_time = time.monotonic()


@BaseRateLimiter.register("min_interval_rate_limiter")
class MinIntervalRateLimiter(BaseRateLimiter):
    """Rate limiter with minimum interval between requests and optional max concurrency.

    Enforces a minimum time between the start of consecutive requests (e.g. RPM limit)
    and optionally limits how many requests can run concurrently.
    """

    min_interval: float = Field(..., ge=0)
    max_concurrent: int | None = Field(default=None, ge=1)

    _state: _MinIntervalRateLimiterState = PrivateAttr()

    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[float]:
        """Wait for rate limit, then yields the time waited in seconds."""
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
            yield elapsed_time

    @override
    def initialize_state(self, existing: Self | None = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Initialize state, sharing from an existing instance or creating fresh."""
        if existing is None:
            self._state = _MinIntervalRateLimiterState(self.max_concurrent)
        else:
            self._state = existing._state

    @classmethod
    def from_rpm(
        cls, rpm: int, max_concurrent: int | None = None, id: str | None = None
    ) -> "MinIntervalRateLimiter":
        """Create a rate limiter from requests-per-minute (RPM).

        Parameters
        ----------
        rpm : int
            Maximum requests per minute. Must be greater than 0.
        max_concurrent : int or None, optional
            Maximum concurrent requests allowed, or None for no limit.
        id : str or None, optional
            Unique identifier. Auto-generated if not provided.

        Returns
        -------
        MinIntervalRateLimiter
            Configured for the given RPM and concurrency.

        Raises
        ------
        ValueError
            If rpm is less than or equal to 0.
        """
        if rpm <= 0:
            raise ValueError("RPM must be greater than 0")

        return cls(
            min_interval=60.0 / rpm,
            max_concurrent=max_concurrent,
            id=id or str(uuid.uuid4()),
        )
