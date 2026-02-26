"""Rate limiter base classes for throttling async operations.

This module provides the abstract base class and registry for rate limiters
that control the frequency and concurrency of async operations (e.g. API calls).
"""

import operator
import os
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Self, override
from weakref import WeakSet

from pydantic import (
    ConfigDict,
    Field,
)

from ..discriminated import Discriminated, discriminated_base

GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS = os.environ.get(
    "GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS", ""
).lower() in ("true", "1", "yes")


class RateLimiterRegistry:
    """Share limiter state across instances despite serialization round-trips.

    When a rate limiter is created (including after deserialization), instances
    with the same id and model fields share the same internal state, so
    throttling is consistent across serialization boundaries.
    """

    _lock: threading.Lock
    _instances: dict[str, WeakSet["BaseRateLimiter[Any]"]]

    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {}

    def register_instance(self, rate_limiter: "BaseRateLimiter") -> None:
        """Register a rate limiter instance, initializing its state.

        Finds an existing instance with the same id and model fields to share
        state with, or lets the limiter create fresh state.

        Parameters
        ----------
        rate_limiter : BaseRateLimiter
            The rate limiter instance to register.
        """
        with self._lock:
            instances = self._instances.get(rate_limiter.id)
            if instances is None:
                instances = WeakSet["BaseRateLimiter"]()
                self._instances[rate_limiter.id] = instances

            all_instances = list(instances)
            matching_instances = [
                instance for instance in all_instances if instance == rate_limiter
            ]

            match = matching_instances[0] if matching_instances else None
            rate_limiter.initialize_state(match)

            if (
                not GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS
                and not match
                and all_instances
            ):
                warnings.warn(
                    (
                        f"Rate limiter with id '{rate_limiter.id}' already registered, "
                        f"this will make BaseRateLimiter.from_id('{rate_limiter.id}') unreliable. "
                        "Set GISKARD_DISABLE_DUPLICATE_RATE_LIMITERS_WARNINGS=1 to disable this warning"
                    ),
                    RuntimeWarning,
                )

            instances.add(rate_limiter)

    def get_instance(self, id: str) -> "BaseRateLimiter[Any]":
        """Retrieve a registered rate limiter by id.

        Parameters
        ----------
        id : str
            The unique identifier of the rate limiter.

        Returns
        -------
        BaseRateLimiter
            The registered rate limiter instance.

        Raises
        ------
        ValueError
            If no rate limiter with the given id is registered.
        """
        with self._lock:
            instances = self._instances.get(id)
            if not instances:
                raise ValueError(f"Rate limiter with id '{id}' not found")

            return next(iter(instances))


@discriminated_base
class BaseRateLimiter(Discriminated, ABC):
    """Abstract base class for rate limiters that throttle async operations.

    Subclasses must implement throttle() and optionally override initialize_state().
    Instances with the same id and model fields share throttling state via the registry.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    _registry: ClassVar[RateLimiterRegistry] = RateLimiterRegistry()

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @override
    def model_post_init(self, context: Any, /) -> None:
        self._registry.register_instance(self)
        super().model_post_init(context)

    @asynccontextmanager
    @abstractmethod
    async def throttle(self) -> AsyncGenerator[float]:
        """Async context manager that enforces rate limiting before yielding.

        Yields
        ------
        float
            The time waited (in seconds) before the operation was allowed to proceed,
            or 0.0 if no wait was required.
        """
        raise NotImplementedError
        yield 0.0  # unreachable; makes this an async generator for @asynccontextmanager typing

    def initialize_state(self, existing: Self | None = None) -> None:
        """Initialize internal state, optionally sharing from an existing matching instance.

        Parameters
        ----------
        existing : Self or None, optional
            An existing matching instance to share state from. If None, fresh
            state should be created.
        """

    @classmethod
    def from_id(cls, id: str) -> "BaseRateLimiter":
        """Retrieve a previously registered rate limiter by id.

        Parameters
        ----------
        id : str
            The unique identifier of the rate limiter.

        Returns
        -------
        BaseRateLimiter
            The registered rate limiter instance.

        Raises
        ------
        ValueError
            If no rate limiter with the given id is registered.
        """
        return cls._registry.get_instance(id)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare rate limiters by model fields only, ignoring internal state.

        This is needed for the registry: when registering, we find matching instances
        by config; the new instance does not have _state set yet.
        """
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
