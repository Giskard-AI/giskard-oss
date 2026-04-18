from .interaction import Interaction
from .trace import Trace

__all__ = ["InteractionSpec", "Interact", "Interaction", "Trace"]


def __getattr__(name: str):
    if name == "InteractionSpec":
        from .base import InteractionSpec

        return InteractionSpec
    if name == "Interact":
        from .interact import Interact

        return Interact
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
