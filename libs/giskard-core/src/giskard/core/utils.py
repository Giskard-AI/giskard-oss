"""Utility constants and helpers for the Giskard library ecosystem."""

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel

GISKARD_LIBS = frozenset(
    [
        "giskard-core",
        "giskard-checks",
        "giskard-scan",
        "giskard-agents",
        "giskard-llm",
    ]
)


class NotProvided(BaseModel):
    """Sentinel class to indicate that a value was not provided."""

    __type__: Literal["not_provided"] = "not_provided"


NOT_PROVIDED = NotProvided()


def provide_not_none[T](value: T | None) -> T | NotProvided:
    return value if value is not None else NOT_PROVIDED


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
