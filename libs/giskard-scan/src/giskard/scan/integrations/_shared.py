"""Helpers shared across third-party scanner adapters."""

from typing import Any


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
