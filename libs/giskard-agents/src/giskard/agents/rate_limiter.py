import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, PrivateAttr, model_validator


class RateLimiterStrategy(BaseModel):
    min_interval: float
    max_concurrent: int = Field(default=5)


class RateLimiter(BaseModel, frozen=True):
    rate_limiter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: RateLimiterStrategy

    _next_request_time: float = PrivateAttr(default_factory=time.monotonic)
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _lock: asyncio.Lock = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._semaphore = asyncio.Semaphore(self.strategy.max_concurrent)
        self._lock = asyncio.Lock()

        _register_rate_limiter(self)

    @classmethod
    def from_rpm(
        cls, rpm: int, max_concurrent: int = 5, rate_limiter_id: str | None = None
    ) -> "RateLimiter":
        """Create a RateLimiter with requests per minute.

        Parameters
        ----------
        rpm : int
            Requests per minute
        max_concurrent : int
            Maximum concurrent requests
        rate_limiter_id : str | None
            Optional rate limiter ID
        """
        min_interval = 60.0 / rpm
        strategy = RateLimiterStrategy(
            min_interval=min_interval, max_concurrent=max_concurrent
        )
        return cls(
            strategy=strategy,
            rate_limiter_id=rate_limiter_id or str(uuid.uuid4()),
        )

    @asynccontextmanager
    async def throttle(self):
        """Acquire rate limiter using async context manager.

        This enforces:
        1. Maximum concurrent requests via semaphore
        2. Minimum interval between request starts
        """
        try:
            await self.acquire()
            yield self
        finally:
            self.release()

    async def acquire(self) -> None:
        # Acquire semaphore to limit concurrency
        await self._semaphore.acquire()

        # Then enforce timing constraints (throttling)
        async with self._lock:
            now = time.monotonic()
            if self._next_request_time > now:
                wait_time = self._next_request_time - now
                await asyncio.sleep(wait_time)
                self._next_request_time += self.strategy.min_interval
            else:
                self._next_request_time = now + self.strategy.min_interval

    def release(self) -> None:
        self._semaphore.release()

    @model_validator(mode="wrap")
    @classmethod
    def _validate_singleton(cls, v, handler, _info) -> "RateLimiter":
        # Determine the rate limiter ID from the input
        rate_limiter_id = None
        if isinstance(v, dict):
            rate_limiter_id = v.get("rate_limiter_id")
        elif isinstance(v, RateLimiter):
            rate_limiter_id = v.rate_limiter_id

        # If ID exists in our global registry, return the singleton
        if rate_limiter_id and rate_limiter_id in _rate_limiters:
            return _rate_limiters[rate_limiter_id]

        # 3. Otherwise, proceed with standard validation/creation
        instance = handler(v)
        return instance

    def __deepcopy__(self, memo) -> "RateLimiter":
        # RateLimiter is a shared resource, so we can just return the same instance.
        return self


_rate_limiters: dict[str, RateLimiter] = {}


def _register_rate_limiter(rate_limiter: RateLimiter) -> None:
    _rate_limiters[rate_limiter.rate_limiter_id] = rate_limiter


def get_rate_limiter(
    rate_limiter_id: str,
) -> RateLimiter:
    """Get or create a rate limiter.

    Parameters
    ----------
    rate_limiter_id : str
        The rate limiter ID to retrieve.

    Returns
    -------
    RateLimiter
        The rate limiter.
    """
    try:
        return _rate_limiters[rate_limiter_id]
    except KeyError as err:
        raise ValueError(f"Rate limiter with id {rate_limiter_id} not found") from err
