"""Helpers shared across third-party scanner adapters.

This module must not import any third-party scanner (garak, deepteam, lidar): it
is imported by every integration, and each scanner is an optional dependency.
Keeping it import-free is what lets the helpers below actually be shared -- a
helper parked next to a ``import garak`` is unreachable from deepteam, which is
how ``await_on_loop`` previously ended up copy-pasted (and drifting) in two
integrations.
"""

import asyncio
from collections.abc import Coroutine, Iterable
from typing import Any, Protocol

from giskard.checks import Interaction, Trace


class RoleContentTurn(Protocol):
    """A chat turn as every scanner models it: a role plus its text content."""

    @property
    def role(self) -> str: ...

    @property
    def content(self) -> Any: ...


async def trace_from_role_content_turns(turns: Iterable[RoleContentTurn]) -> Trace:  # pyright: ignore[reportMissingTypeArgument]
    """Rebuild a display Trace from a scanner's flat list of chat turns.

    Used when a scanner hands back a finished conversation instead of driving our
    bridge turn by turn, so no lossless typed Trace was captured. Each ``user``
    turn is paired with the ``assistant`` reply that follows it; ``system``/
    ``tool`` turns carry no input/output pair and are skipped, and a trailing
    unanswered user turn yields ``outputs=None``.
    """
    interactions: list[Interaction] = []  # pyright: ignore[reportMissingTypeArgument]
    pending_input: Any = None
    for turn in turns:
        if turn.role == "user":
            if pending_input is not None:
                interactions.append(Interaction(inputs=pending_input, outputs=None))
            pending_input = turn.content
        elif turn.role == "assistant" and pending_input is not None:
            interactions.append(Interaction(inputs=pending_input, outputs=turn.content))
            pending_input = None
    if pending_input is not None:
        interactions.append(Interaction(inputs=pending_input, outputs=None))
    return await Trace.from_interactions(*interactions)


def await_on_loop[T](
    coro: Coroutine[Any, Any, T],
    loop: asyncio.AbstractEventLoop | None,
) -> T:
    """Run *coro* without spawning a nested event loop in a worker thread.

    Scanners call our generator/detector/judge code synchronously, often from a
    worker thread (garak probes, deepteam's sync ``generate``). ``asyncio.run``
    there creates an ephemeral loop, which breaks the ``asyncio.Lock`` shared by
    ``DatasetInputGenerator`` and the Giskard generator when probes run in
    parallel. When *loop* is the scan's running loop, schedule the coroutine on
    it via ``run_coroutine_threadsafe`` instead.
    """
    if loop is None:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def reject_unexpected_kwargs(tool: str, kwargs: dict[str, Any]) -> None:
    """Raise ``TypeError`` if ``kwargs`` still holds keys after the adapter popped
    the options it recognizes.

    Each adapter knows its own valid kwargs and pops them; anything left over is a
    caller typo (e.g. ``probe`` for ``probes``) that would otherwise be silently
    dropped. The message names the public ``third_party_scan(tool=...)`` entry point
    the caller actually used, so it stays useful even though validation happens in
    the adapter.
    """
    if kwargs:
        unexpected = ", ".join(repr(key) for key in kwargs)
        raise TypeError(
            f"third_party_scan(tool={tool!r}) got unexpected keyword "
            f"argument(s): {unexpected}"
        )
