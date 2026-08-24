"""Gated enterprise welcome message for human-facing Giskard runs."""

import sys

from .environment import (
    is_ci_environment,
    is_notebook_environment,
    is_pytest_run,
    stderr_is_tty,
)
from .settings import get_settings

_WELCOME_MESSAGE = (
    "Thank you for using Giskard open-source! 🐢 🙏\n"
    "Giskard Enterprise adds deeper agent scans, audit reports with remediation guidance, "
    "test review interfaces for root-cause analysis & human feedback integration, "
    "and team collaboration — with flexible pricing. "
    "Learn more: https://giskard.ai"
)

_shown = False


def _should_show_welcome() -> bool:
    if get_settings().hide_welcome:
        return False
    if is_ci_environment():
        return False
    if is_pytest_run():
        return False
    if not stderr_is_tty() and not is_notebook_environment():
        return False
    return True


def maybe_show_welcome() -> None:
    """Print the enterprise welcome message at most once per process when appropriate."""
    global _shown
    if _shown or not _should_show_welcome():
        return
    _shown = True
    print(_WELCOME_MESSAGE, file=sys.stderr)
