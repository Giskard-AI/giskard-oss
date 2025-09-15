"""Core exports: base check and interaction types."""

from .check import Check, CheckResult, CheckStatus, list_registered_check_kinds
from .interactions import Interaction, list_registered_interaction_kinds
from .registry import DuplicateKindError, Registry, UnknownKindError

__all__ = [
    "Interaction",
    "Check",
    "CheckResult",
    "CheckStatus",
    "Registry",
    "UnknownKindError",
    "DuplicateKindError",
    "list_registered_check_kinds",
    "list_registered_interaction_kinds",
]
