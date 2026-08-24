"""Runtime environment detection shared across giskard-core."""

import os
import sys
from typing import Literal

EnvironmentKind = Literal["ci", "colab", "kaggle", "local"]

TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "t", "y"})


def is_truthy_env(value: str | None) -> bool:
    """Return whether an environment variable value should be treated as true."""
    if value is None:
        return False

    return value.strip().lower() in TRUTHY_ENV_VALUES


def is_ci_environment() -> bool:
    """Return whether the current process is running in a CI environment."""
    return is_truthy_env(os.getenv("CI")) or is_truthy_env(os.getenv("TF_BUILD"))


def is_pytest_run() -> bool:
    """Return whether pytest is executing the current process."""
    return os.getenv("PYTEST_VERSION") is not None


def is_colab_environment() -> bool:
    """Return whether the process is running inside Google Colab."""
    return "google.colab" in sys.modules


def is_kaggle_environment() -> bool:
    """Return whether the process is running inside a Kaggle kernel."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def is_ipython_environment() -> bool:
    """Return whether IPython is loaded (Jupyter and similar notebook shells)."""
    return "IPython" in sys.modules


def is_notebook_environment() -> bool:
    """Return whether the process is running in a common notebook environment."""
    return is_colab_environment() or is_kaggle_environment() or is_ipython_environment()


def classify_environment() -> EnvironmentKind:
    """Classify the coarse runtime environment for telemetry and UX gating."""
    if is_ci_environment():
        return "ci"
    if is_colab_environment():
        return "colab"
    if is_kaggle_environment():
        return "kaggle"
    return "local"


def stderr_is_tty() -> bool:
    """Return whether stderr is connected to an interactive terminal."""
    stderr = sys.stderr
    isatty = getattr(stderr, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (AttributeError, ValueError, OSError):
        return False
