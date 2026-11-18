"""Route-gated reasoning replay (ADR-0003)."""

from collections.abc import Iterable

from ..types import ReasoningContent


def replayable[T](items: Iterable[T], provider: str) -> list[T]:
    """Drop reasoning that did not originate from *provider*.

    A ``ReasoningContent`` is kept only when its ``provider`` equals *provider*
    exactly; ``provider=None`` never matches. Routes of the same family are not
    interchangeable (``openai`` reasoning is dropped on ``azure``). Every other
    item is kept, in order.
    """
    return [
        item
        for item in items
        if not isinstance(item, ReasoningContent) or item.provider == provider
    ]
