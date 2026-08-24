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
    return (
        not get_settings().hide_welcome
        and not is_ci_environment()
        and not is_pytest_run()
        and (stderr_is_tty() or is_notebook_environment())
    )


def maybe_show_welcome() -> None:
    """Print the enterprise welcome message at most once per process when appropriate."""
    global _shown
    try:
        if _shown or not _should_show_welcome():
            return
        _shown = True
        print(_WELCOME_MESSAGE, file=sys.stderr)
    except Exception:
        # Best-effort UX: never abort a suite or scan because the banner failed.
        return
