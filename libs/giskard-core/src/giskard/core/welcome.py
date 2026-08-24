"""Gated enterprise welcome message for human-facing Giskard runs."""

import os
import sys

_HIDE_WELCOME_ENV_VAR = "GISKARD_HIDE_WELCOME"
_TRUTHY_VALUES = {"1", "true", "yes", "on", "t", "y"}

_WELCOME_MESSAGE = (
    "Thank you for using Giskard open-source! 🐢 🙏\n"
    "Giskard Enterprise adds deeper agent scans, audit reports with remediation guidance, "
    "test review interfaces for root-cause analysis & human feedback integration, "
    "and team collaboration — with flexible pricing. "
    "Learn more: https://giskard.ai"
)

_shown = False


def _is_true_str(value: str | None) -> bool:
    if value is None:
        return False

    return value.strip().lower() in _TRUTHY_VALUES


def _is_ci() -> bool:
    return _is_true_str(os.getenv("CI")) or _is_true_str(os.getenv("TF_BUILD"))


def _is_pytest() -> bool:
    return os.getenv("PYTEST_VERSION") is not None


def _is_notebook() -> bool:
    if "google.colab" in sys.modules:
        return True
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
        return True
    return "IPython" in sys.modules


def _stderr_is_tty() -> bool:
    stderr = sys.stderr
    isatty = getattr(stderr, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _should_show_welcome() -> bool:
    if _is_true_str(os.getenv(_HIDE_WELCOME_ENV_VAR)):
        return False
    if _is_ci():
        return False
    if _is_pytest():
        return False
    if not _stderr_is_tty() and not _is_notebook():
        return False
    return True


def maybe_show_welcome() -> None:
    """Print the enterprise welcome message at most once per process when appropriate."""
    global _shown
    if _shown or not _should_show_welcome():
        return
    _shown = True
    print(_WELCOME_MESSAGE, file=sys.stderr)
