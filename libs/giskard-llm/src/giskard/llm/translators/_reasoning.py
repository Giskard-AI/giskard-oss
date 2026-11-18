"""Route-gated reasoning replay (ADR-0003)."""

from collections.abc import Iterable
from typing import TypeVar

from ..types import ReasoningContent, ResponseReasoningItem

T = TypeVar("T")


def replayable(items: Iterable[T], provider: str) -> list[T]:
    """Drop reasoning that did not originate from *provider*.

    A ``ReasoningContent`` is kept only when its ``provider`` equals *provider*
    exactly; ``provider=None`` never matches. Routes of the same family are not
    interchangeable (``openai`` reasoning is dropped on ``azure``). Every other
    item is kept, in order.
    """
    return [item for item in items if _replays_on(item, provider)]


def _replays_on(item: object, provider: str) -> bool:
    if isinstance(item, ReasoningContent):
        return item.provider == provider
    # Legacy untagged item: it has no Originating Provider, so it never replays.
    return not isinstance(item, ResponseReasoningItem)
