import sys
from os import getenv

from giskard.core.telemetry.telemetry import _is_true_str

_WELCOME_MESSAGE = (
    "Thank you for using Giskard open-source! 🐢 🙏\n"
    "Giskard Enterprise adds deeper agent scans, audit reports with remediation guidance, "
    "test review interfaces for root-cause analysis & human feedback integration, "
    "and team collaboration — with flexible pricing. "
    "Learn more: https://giskard.ai"
)

_shown = False


def _should_show_welcome() -> bool:
    value = getenv("GISKARD_QUIET")
    return not _is_true_str(value)


def maybe_show_welcome() -> None:
    """Print the enterprise welcome message at most once per process."""
    global _shown
    if _shown or not _should_show_welcome():
        return
    _shown = True
    print(_WELCOME_MESSAGE, file=sys.stderr)
