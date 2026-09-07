"""Utility constants and helpers for the Giskard library ecosystem."""

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version

GISKARD_LIBS = frozenset(
    [
        "giskard-core",
        "giskard-checks",
        "giskard-scan",
        "giskard-agents",
        "giskard-llm",
    ]
)

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "t", "y"})


def is_true_env_str(value: str | None) -> bool:
    """Return whether an environment-variable value represents true.

    Parameters
    ----------
    value : str or None
        Environment-variable value to parse.
    """
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


def get_lib_version(lib: str, default: str = "unknown") -> str:
    try:
        return version(lib)
    except PackageNotFoundError:
        return default


def _get_libs_version(
    libs: Iterable[str], /, default: str = "unknown"
) -> dict[str, str]:
    return {lib: get_lib_version(lib, default) for lib in libs}


GISKARD_LIBS_VERSIONS = _get_libs_version(GISKARD_LIBS, "not_installed")
