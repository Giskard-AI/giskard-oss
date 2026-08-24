"""One-time enterprise welcome message for giskard-checks imports."""

import sys
from os import getenv

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "t", "y"})

_WELCOME_MESSAGE = (
    "Thank you for using Giskard open-source! 🐢 🙏\n"
    "Giskard Enterprise adds deeper agent scans, audit reports with remediation guidance, "
    "test review interfaces for root-cause analysis & human feedback integration, "
    "and team collaboration — with flexible pricing. "
    "Learn more: https://giskard.ai"
)

_shown = False


def _should_show_welcome() -> bool:
    value = getenv("GISKARD_HIDE_WELCOME")
    return value is None or value.strip().lower() not in _TRUTHY_ENV_VALUES


def maybe_show_welcome() -> None:
    """Print the enterprise welcome message at most once per process."""
    global _shown
    try:
        if _shown or not _should_show_welcome():
            return
        _shown = True
        print(_WELCOME_MESSAGE, file=sys.stderr)
    except Exception:
        # Best-effort UX: never abort a suite or scan because the banner failed.
        return
