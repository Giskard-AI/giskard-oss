"""Public package exports for giskard.checks.

This package provides primitives to model interactions, implement checks, and
run them via a lightweight test runner. The most commonly used subpackages are
exposed as modules. Import specific names from their respective subpackages.
"""

# Re-export subpackages as modules for convenient access, without star imports
from pathlib import Path

from counterpoint import add_prompts_path

from . import checks as checks
from . import core as core
from . import generators as generators
from . import testing as testing
from .settings import get_default_generator, set_default_generator

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.checks")

__all__ = [
    "core",
    "generators",
    "testing",
    "checks",
    "set_default_generator",
    "get_default_generator",
]
