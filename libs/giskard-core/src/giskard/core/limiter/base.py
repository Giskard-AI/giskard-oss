import operator
import os
import threading
import uuid
import warnings
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

GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS = os.environ.get(
    "GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS", ""
).lower() in ("true", "1", "yes")


class RateLimiterRegistry:
    """Share limiter state across instances despite serialization round-trips."""

    _lock: threading.Lock
    # WeakSet is used to avoid strong references to the instances, allowing them to be garbage collected
    _instances: dict[str, WeakSet["BaseRateLimiter[Any]"]]

    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {}

    def register_instance(self, rate_limiter: "BaseRateLimiter"):
        """Register a rate limiter instance and share state with compatible existing ones.

        When a rate limiter is created (including after deserialization), we look for
        an existing instance with the same id and model fields. If found, we reuse its
        _state so throttling is shared across instances (e.g. before/after serialization).
        """
        with self._lock:
            instances = self._instances.get(rate_limiter.id)
            if instances is None:
                instances = WeakSet["BaseRateLimiter"]()
                self._instances[rate_limiter.id] = instances

            # Use a list to ensure nothing gets removed during iteration
            all_instances = list(instances)
            # Match by model fields only; existing instances have _state, new one does not.
            matching_instances = [
                instance for instance in all_instances if instance == rate_limiter
            ]

            if matching_instances:
                # Reuse existing state (e.g. deserialized instance joins existing).
                rate_limiter._state = matching_instances[0]._state
                return

            # Same id but different config: from_id() would be ambiguous.
            if not GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS and len(
                all_instances
            ) > len(matching_instances):
                warnings.warn(
                    (
                        f"Rate limiter with id '{rate_limiter.id}' already registered,"
                        f"this will make RateLimiter.from_id('{rate_limiter.id}') ambiguous."
                        "Set GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS=1 to disable this warning"
                    ),
                    RuntimeWarning,
                )

            # No match: create fresh state and register.
            rate_limiter._state = rate_limiter.create_initial_state()
            instances.add(rate_limiter)

    def get_instance(self, id: str) -> "BaseRateLimiter[Any]":
        instances = self._instances.get(id)
        if instances is None:
            raise ValueError(f"Rate limiter with id '{id}' not found")

        return next(iter(instances))


@discriminated_base
class BaseRateLimiter(Discriminated):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    _registry: ClassVar[RateLimiterRegistry] = RateLimiterRegistry()

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

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

    def __eq__(self, other):
        # By default Pydantic includes private attributes in equality comparison;
        # we override to compare only model fields, ignoring _state. This is needed
        # for the registry mechanism: in register_instance we find matching instances,
        # which current instance does not have _state set yet.
        if not isinstance(other, type(self)):
            return False

        model_fields = type(self).model_fields
        if not model_fields:
            return True

        getter = operator.itemgetter(*model_fields)
        try:
            return getter(self.__dict__) == getter(other.__dict__)
        except KeyError:
            return False
