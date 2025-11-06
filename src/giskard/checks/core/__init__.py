"""Core exports: base check and interaction types."""

from .check import Check, CheckResult, CheckStatus
from .context import Context
from .interaction import Interaction

__all__ = [
    "Interaction",
    "Check",
    "CheckResult",
    "CheckStatus",
    "Context",
]
