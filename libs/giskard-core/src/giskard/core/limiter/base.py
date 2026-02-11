import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, override
from weakref import WeakSet

from pydantic import (
    ConfigDict,
    Field,
)

from ..discriminated import Discriminated, discriminated_base


class RateLimiterRegistry:
    """Share limiter state across instances despite serialization round-trips."""

    _lock: threading.Lock = threading.Lock()
    _instances: dict[str, WeakSet["BaseRateLimiter"]]
    _states: dict[str, Any]

    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {}
        self._states = {}

    def register_instance(self, rate_limiter: "BaseRateLimiter"):
        with self._lock:
            instances = self._instances.get(rate_limiter.id)
            if instances is None:
                instances = WeakSet()
                self._instances[rate_limiter.id] = instances

            if instances:
                existing_instance = next(iter(instances))
                if existing_instance != rate_limiter:
                    raise ValueError(
                        f"Rate limiter with id '{rate_limiter.id}' already registered"
                    )
            else:
                # Initialize the state for the first instance
                self._states[rate_limiter.id] = rate_limiter.create_initial_state()

            instances.add(rate_limiter)

    def get_state(self, rate_limiter: "BaseRateLimiter") -> Any:
        return self._states.get(rate_limiter.id, None)

    def get_instance(self, id: str) -> "BaseRateLimiter":
        instances = self._instances.get(id)
        if instances is None:
            raise ValueError(f"Rate limiter with id '{id}' not found")

        return next(iter(instances))


@discriminated_base
class BaseRateLimiter(Discriminated):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _registry: ClassVar[RateLimiterRegistry] = RateLimiterRegistry()

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @override
    def model_post_init(self, context: Any, /) -> None:
        self._registry.register_instance(
            self,
        )
        super().model_post_init(context)

    @asynccontextmanager
    def throttle(self) -> AsyncGenerator[float]:
        raise NotImplementedError

    def create_initial_state(self) -> Any:
        return None

    @property
    def state(self) -> Any:
        return self._registry.get_state(self)

    @classmethod
    def from_id(cls, id: str) -> "BaseRateLimiter":
        return cls._registry.get_instance(id)
