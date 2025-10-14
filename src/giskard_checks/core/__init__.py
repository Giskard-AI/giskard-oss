"""Core exports: base check and interaction types."""

from .check import Check, CheckResult, CheckStatus
from .interactions import Interaction

__all__ = [
    "Interaction",
    "Check",
    "CheckResult",
    "CheckStatus",
]
