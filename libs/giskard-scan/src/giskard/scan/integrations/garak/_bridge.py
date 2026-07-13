import asyncio
from collections.abc import Coroutine
from typing import Any

from garak.attempt import Conversation


def _await_on_loop[T](
    coro: Coroutine[Any, Any, T],
    loop: asyncio.AbstractEventLoop | None,
) -> T:
    """Run *coro* without spawning a nested event loop in a worker thread.

    Garak calls generator/detector code synchronously, often from
    ``asyncio.to_thread`` workers. ``asyncio.run`` there creates ephemeral loops
    that break ``asyncio.Lock`` in ``DatasetInputGenerator`` when probes run in
    parallel. When *loop* is the scan's running loop, schedule the coroutine there
    via ``run_coroutine_threadsafe`` instead.
    """
    if loop is None:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _conv_uuid(conversation: Conversation) -> str | None:
    for turn in conversation.turns:
        notes = turn.content.notes
        if notes and notes.get("uuid"):
            return notes["uuid"]
    return None
