"""Public package exports for giskard.checks."""

from pathlib import Path

from giskard.agents import add_prompts_path

from . import builtin
from .settings import get_default_generator, set_default_generator

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.checks")


__all__ = [
    "builtin",
    "set_default_generator",
    "get_default_generator",
]
