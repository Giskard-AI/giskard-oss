"""Public package exports for giskard_checks.

This package provides primitives to model interactions, implement checks, and
run them via a lightweight test runner. The most commonly used subpackages are
exposed as modules. Import specific names from their respective subpackages.
"""

# Re-export subpackages as modules for convenient access, without star imports
from pathlib import Path
from typing import TYPE_CHECKING

from counterpoint import add_prompts_path

from . import checks as checks
from . import core as core
from . import testing as testing

if TYPE_CHECKING:
    from counterpoint.generators.base import BaseGenerator

# Global default generator
_default_generator: "BaseGenerator | None" = None


def set_default_generator(generator: "BaseGenerator") -> None:
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator
        The counterpoint generator to use as default for all LLM checks.
    """
    global _default_generator
    _default_generator = generator


def get_default_generator() -> "BaseGenerator | None":
    """Get the current default generator.

    Returns
    -------
    BaseGenerator | None
        The current default generator, or None if none has been set.
    """
    return _default_generator


add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard_checks")

__all__ = [
    "core",
    "testing",
    "checks",
    "set_default_generator",
    "get_default_generator",
]
