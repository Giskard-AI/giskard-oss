import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, override
from weakref import WeakSet

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
)

from ..discriminated import Discriminated, discriminated_base


class RateLimiterRegistry:
    """Share limiter state across instances despite serialization round-trips."""

    _lock: threading.Lock
    _instances: dict[str, WeakSet["BaseRateLimiter[Any]"]]

    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {}

    def register_instance(self, rate_limiter: "BaseRateLimiter"):
        with self._lock:
            instances = self._instances.get(rate_limiter.id)
            if instances is None:
                instances = WeakSet["BaseRateLimiter"]()
                self._instances[rate_limiter.id] = instances

            if instances:
                try:
                    existing_instance = next(iter(instances))
                    rate_limiter._state = (
                        existing_instance._state
                    )  # Set state to ensure equality check works
                    if existing_instance != rate_limiter:
                        raise ValueError(
                            f"Rate limiter with id '{rate_limiter.id}' already registered"
                        )

                    instances.add(rate_limiter)
                    return
                except StopIteration:
                    pass  # last instance was deleted by gc

            rate_limiter._state = rate_limiter.create_initial_state()
            instances.add(rate_limiter)

    def get_instance(self, id: str) -> "BaseRateLimiter[Any]":
        instances = self._instances.get(id)
        if instances is None:
            raise ValueError(f"Rate limiter with id '{id}' not found")

        return next(iter(instances))


@discriminated_base
class BaseRateLimiter(Discriminated):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _registry: ClassVar[RateLimiterRegistry] = RateLimiterRegistry()

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    _state: Any = PrivateAttr()

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

    @classmethod
    def from_id(cls, id: str) -> "BaseRateLimiter":
        return cls._registry.get_instance(id)
