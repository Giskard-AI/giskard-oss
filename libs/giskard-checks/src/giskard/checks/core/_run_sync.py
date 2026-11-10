"""Shared support for synchronous wrappers around async run methods."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


def run_sync[**P, T](
    run: Callable[P, Coroutine[Any, Any, T]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run an async callable when the current thread has no active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run(*args, **kwargs))

    raise RuntimeError(
        "run_sync() cannot be called while an asyncio event loop is running; "
        "use await obj.run(...) instead."
    )
