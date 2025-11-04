"""Core exports: base check and interaction result types."""

from .check import Check, CheckResult, CheckStatus
from .context import Context
from .interaction_result import InteractionResult

__all__ = [
    "InteractionResult",
    "Check",
    "CheckResult",
    "CheckStatus",
    "Context",
]
