from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .interaction import Trace


class InputGenerationException(Exception):
    """Raised when an input generator cannot produce a valid input (e.g. schema incompatibility)."""


class InteractionGenerationError(Exception):
    """Raised when an interaction generator fails, carrying the progress made so far.

    The error that stopped the generator is chained as `__cause__`.

    Attributes
    ----------
    partial_trace : Trace[Any, Any]
        Trace containing every interaction completed before the failure.
    """

    def __init__(self, partial_trace: "Trace[Any, Any]"):
        super().__init__("Interaction generation failed")
        self.partial_trace = partial_trace
