"""Core shared utilities and foundational components for the Giskard library ecosystem.

This package provides minimal, essential building blocks that are shared across
all Giskard packages, including discriminated unions, error handling, type
definitions, configuration patterns, and serialization utilities.
"""

import importlib.util

from .discriminated import Discriminated, discriminated_base
from .errors import Error

LEGACY_GISKARD_PACKAGE_NAME = "giskard"

spec = importlib.util.find_spec(LEGACY_GISKARD_PACKAGE_NAME)

# Check if the legacy package is installed
if spec and spec.has_location:
    raise ImportError(
        (
            f"Package conflict detected: The legacy package '{LEGACY_GISKARD_PACKAGE_NAME}' is installed "
            "and conflicts with the new namespace structure provided by 'giskard-core'.\n\n"
            "To resolve this issue, please uninstall the legacy package "
            f"by running: pip uninstall {LEGACY_GISKARD_PACKAGE_NAME}"
        )
    )

__all__ = [
    # Discriminated unions
    "Discriminated",
    "discriminated_base",
    # Error handling
    "Error",
]
