import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..discriminated import Discriminated, discriminated_base


class ThrottleEvent(BaseModel, frozen=True):
    rate_limiter_id: str
    rule: "RateLimiterRule[Any]"
    waited_time: float


class ThrottleEvents(BaseModel, frozen=True):
    events: list[ThrottleEvent]

    @property
    def waited_time(self) -> float:
        return sum(event.waited_time for event in self.events)

    def monitor(
        self, rate_limiter_id: str, rule: "RateLimiterRule[Any]", waited_time: float
    ):
        if waited_time < 1e-3:
            return self

        return self.model_copy(
            update={
                "events": self.events
                + [
                    ThrottleEvent(
                        rate_limiter_id=rate_limiter_id,
                        rule=rule,
                        waited_time=waited_time,
                    )
                ]
            }
        )

    def __add__(self, other: "ThrottleEvents") -> "ThrottleEvents":
        return self.model_copy(update={"events": self.events + other.events})


class _RateLimiterRegistry:
    _lock: threading.Lock = threading.Lock()
    _registered_ids: dict[str, tuple["RateLimiterRule[Any]", ...]] = {}
    _rate_limiter_rules_states: dict[tuple[str, "RateLimiterRule[Any]"], Any] = {}

    def register_rate_limiter(self, rate_limiter: "RateLimiter"):
        registered_rules = self._registered_ids.get(rate_limiter.id, None)

        if registered_rules is not None:
            if registered_rules != rate_limiter.rules:
                raise ValueError(
                    f"Rate limiter with id '{rate_limiter.id}' already registered with different rules"
                )
            return

        with self._lock:
            registered_rules = self._registered_ids.get(rate_limiter.id, None)
            if registered_rules is not None:
                if registered_rules != rate_limiter.rules:
                    raise ValueError(
                        f"Rate limiter with id '{rate_limiter.id}' already registered with different rules"
                    )
                return

            self._registered_ids[rate_limiter.id] = rate_limiter.rules
            for rule in rate_limiter.rules:
                self._rate_limiter_rules_states[(rate_limiter.id, rule)] = (
                    rule.build_initial_state()
                )

    def get_rate_limiter(self, id: str) -> "RateLimiter":
        rules = self._registered_ids.get(id, None)

        if rules is None:
            raise ValueError(f"Rate limiter with id '{id}' not found")

        return RateLimiter(id=id, rules=rules)

    def get_rate_limiter_rules_state(
        self, rate_limiter: "RateLimiter", rule: "RateLimiterRule[Any]"
    ) -> Any:
        return self._rate_limiter_rules_states[(rate_limiter.id, rule)]


@discriminated_base
class RateLimiterRule[T](Discriminated):
    """A rule for a rate limiter."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @asynccontextmanager
    def throttle(self, state: T) -> AsyncGenerator[None]:
        raise NotImplementedError

    def build_initial_state(self) -> T:
        raise NotImplementedError


class RateLimiter(BaseModel, frozen=True):
    """Abstract base for rate limiters using async context managers."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rules: tuple[RateLimiterRule[Any], ...] = Field(default_factory=tuple)

    _registry: ClassVar[_RateLimiterRegistry] = _RateLimiterRegistry()

    @classmethod
    def from_id(cls, id: str) -> "RateLimiter":
        return cls._registry.get_rate_limiter(id)

    @classmethod
    def from_rules(
        cls, *rules: RateLimiterRule[Any], id: str | None = None
    ) -> "RateLimiter":
        if id is None:
            id = str(uuid.uuid4())

        return RateLimiter(id=id, rules=rules)

    @classmethod
    def max_concurrent(
        cls, max_concurrent: int, id: str | None = None
    ) -> "RateLimiter":
        from .max_concurrent import MaxConcurrentRequests

        return cls.from_rules(
            MaxConcurrentRequests(max_concurrent=max_concurrent), id=id
        )

    @classmethod
    def min_interval(
        cls,
        min_interval: float,
        max_concurrent: int | None = None,
        id: str | None = None,
    ) -> "RateLimiter":
        from .min_interval import MinInterval

        rules: list[RateLimiterRule[Any]] = []
        if max_concurrent is not None:
            from .max_concurrent import MaxConcurrentRequests

            rules.append(MaxConcurrentRequests(max_concurrent=max_concurrent))

        rules.append(MinInterval(min_interval=min_interval))

        return cls.from_rules(*rules, id=id)

    @classmethod
    def from_rpm(
        cls, rpm: int, max_concurrent: int | None = None, id: str | None = None
    ) -> "RateLimiter":
        if rpm <= 0:
            raise ValueError("RPM must be greater than 0")

        return cls.min_interval(
            min_interval=60.0 / rpm, max_concurrent=max_concurrent, id=id
        )

    def model_post_init(self, context: Any, /) -> None:
        self._registry.register_rate_limiter(self)

    @asynccontextmanager
    async def throttle(
        self,
    ) -> AsyncGenerator[ThrottleEvents, None]:
        throttled_times = NO_THROTTLE
        if not self.rules:
            yield throttled_times
            return

        async with AsyncExitStack() as stack:
            for rule in self.rules:
                state = self._registry.get_rate_limiter_rules_state(self, rule)
                start_time = time.monotonic()
                await stack.enter_async_context(rule.throttle(state))
                end_time = time.monotonic()
                throttled_times = throttled_times.monitor(
                    self.id, rule, end_time - start_time
                )

            yield throttled_times


NO_THROTTLE = ThrottleEvents(events=[])
