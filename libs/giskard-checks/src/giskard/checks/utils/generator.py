"""Utilities for converting values and generators to async generators."""

from collections.abc import AsyncGenerator, Generator
from typing import Any

from ..core.types import SyncOrAsyncGenerator


async def _to_async_generator[YieldType, SendType](
    sync_generator: Generator[YieldType, SendType, None],
) -> AsyncGenerator[YieldType, SendType]:
    """Convert a synchronous generator to an async generator."""

    iterator = iter(sync_generator)

    try:
        next_item = yield next(iterator)
        while True:
            next_item = yield iterator.send(next_item)
    except StopIteration:
        return
    finally:
        iterator.close()


async def _single_value_to_async_generator[YieldType](
    value: YieldType,
) -> AsyncGenerator[YieldType, Any]:
    """Convert a single value to an async generator that yields that value."""
    yield value


def a_generator[YieldType, SendType](
    value_or_generator: YieldType | SyncOrAsyncGenerator[YieldType, SendType],
) -> AsyncGenerator[YieldType, SendType]:
    """Convert a value or generator (sync or async) into an async generator."""
    if isinstance(value_or_generator, AsyncGenerator):
        return value_or_generator
    elif isinstance(value_or_generator, Generator):
        return _to_async_generator(value_or_generator)

    return _single_value_to_async_generator(value_or_generator)
