"""Utilities for converting values and callables to async callables."""

import inspect
from collections.abc import Awaitable, Callable
from typing import cast


def _single_value_wrapper[ParamType, ReturnType](
    value: ReturnType,
) -> Callable[[ParamType], Awaitable[ReturnType]]:
    """Wrap a single value into an async callable that returns that value."""

    async def wrapper(*_args: ParamType, **_kwargs: ParamType) -> ReturnType:
        return value

    return wrapper


def _sync_callable_wrapper[ParamType, ReturnType](
    sync_func: Callable[[ParamType], ReturnType],
) -> Callable[[ParamType], Awaitable[ReturnType]]:
    """Wrap a synchronous callable into an async callable."""

    async def wrapper(*args: ParamType, **kwargs: ParamType) -> ReturnType:
        return sync_func(*args, **kwargs)

    return wrapper


def a_callable[PartialParamType, ParamType, ReturnType](
    value_or_callable: ReturnType
    | Callable[[ParamType], ReturnType]
    | Callable[[ParamType], Awaitable[ReturnType]],
) -> Callable[[ParamType], Awaitable[ReturnType]]:
    """Convert a value or callable (sync or async) into an async callable."""
    if isinstance(value_or_callable, Callable):
        if inspect.iscoroutinefunction(value_or_callable):
            return value_or_callable

        sync_callable = cast(Callable[[ParamType], ReturnType], value_or_callable)
        return _sync_callable_wrapper(sync_callable)

    return _single_value_wrapper(value_or_callable)
