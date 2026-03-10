from contextlib import asynccontextmanager
from typing import Self

from giskard.core import BaseRateLimiter
from pydantic import BaseModel, Field


class WithRateLimiter(BaseModel):
    """Adds a rate limiter to the generator."""

    rate_limiter: BaseRateLimiter | None = Field(default=None)

    def with_rate_limiter(self, rate_limiter: BaseRateLimiter | str | None) -> Self:
        if isinstance(rate_limiter, str):
            rate_limiter = BaseRateLimiter.from_id(rate_limiter)

        return self.model_copy(update={"rate_limiter": rate_limiter})

    @asynccontextmanager
    async def _throttle(self):
        if self.rate_limiter is None:
            yield 0.0
            return

        async with self.rate_limiter.throttle() as waited:
            yield waited
