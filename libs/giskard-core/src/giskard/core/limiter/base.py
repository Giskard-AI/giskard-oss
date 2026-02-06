import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Callable

from pydantic import Field

from ..discriminated import Discriminated, discriminated_base


def compute_waited_time(waited_time: float, threshold: float = 1e-3) -> float:
    if waited_time < threshold:
        return 0
    return waited_time


class RateLimiterStateRegistry[T]:
    """Share limiter state across instances despite serialization round-trips."""

    _lock: asyncio.Lock = asyncio.Lock()
    _shared_state: dict[tuple[str, ...], T] = {}

    async def get_or_create_state(
        self, id: tuple[str, ...], factory: Callable[[], T]
    ) -> T:
        state = self._shared_state.get(id)
        if state is not None:
            return state

        async with self._lock:
            state = self._shared_state.get(id)
            if state is not None:
                return state

            state = factory()
            self._shared_state[id] = state

        return state


@discriminated_base
class RateLimiter(Discriminated):
    """Abstract base for rate limiters using async context managers."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @asynccontextmanager
    def throttle(self) -> AsyncGenerator[float]:
        raise NotImplementedError
