"""Synchronous entry points for the async-first checks API."""

import asyncio
import re
from collections.abc import Callable, Coroutine
from typing import Any


def call_sync[**P, T](
    fn: Callable[P, Coroutine[Any, Any, T]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run the coroutine function ``fn`` to completion from synchronous code.

    Behaves like ``asyncio.run(fn(*args, **kwargs))`` when no event loop is
    running in the current thread (plain scripts, synchronous tests, worker
    threads). When a loop is already running (Jupyter/IPython, async tests,
    async web handlers, ``async def`` functions), it raises instead of nesting,
    patching, or blocking that loop.

    Parameters
    ----------
    fn : Callable[P, Coroutine[Any, Any, T]]
        Coroutine function to run, typically a bound ``run`` method.
    *args : P.args
        Positional arguments forwarded unchanged to ``fn``.
    **kwargs : P.kwargs
        Keyword arguments forwarded unchanged to ``fn``.

    Returns
    -------
    T
        The value returned by the coroutine.

    Raises
    ------
    RuntimeError
        If an asyncio event loop is already running in the current thread.
        ``fn`` is not called in that case, so no coroutine is left un-awaited.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fn(*args, **kwargs))

    bound_to = getattr(fn, "__self__", None)
    if bound_to is not None:
        owner = type(bound_to).__name__
    else:
        owner = getattr(fn, "__qualname__", type(fn).__name__)
    # CamelCase -> snake_case, keeping acronyms together (LLMScenario -> llm_scenario)
    instance = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", owner
    ).lower()
    raise RuntimeError(
        f"{owner}.run_sync() cannot be called while an asyncio event loop is "
        "running (for example in a Jupyter notebook, an async test, or an "
        f"`async def` function). Use `await {instance}.run(...)` instead."
    )
