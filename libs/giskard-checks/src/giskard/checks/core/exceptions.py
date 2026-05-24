"""Custom exceptions for giskard-checks."""

from __future__ import annotations

import importlib
from types import ModuleType


class InputGenerationException(Exception):
    """Raised when an input generator cannot produce a valid input (e.g. schema incompatibility)."""


class OptionalDependencyError(ImportError):
    """Raised when a built-in check needs a package that is not installed.

    Subclasses :class:`ImportError` so callers can catch either type. The
    message always tells the user how to install the missing package via the
    appropriate ``giskard-checks[<extra>]`` pip extra.
    """


def require_optional(
    package: str, extra: str, *, feature: str | None = None
) -> ModuleType:
    """Import an optional dependency or raise :class:`OptionalDependencyError`.

    Built-in checks that depend on libraries outside the core dependency set
    (for example NLP helpers such as ``textblob`` or ``textstat``) should call
    this from inside :meth:`Check.run` to keep construction cheap and to give
    users a uniform install hint when the package is missing.

    Parameters
    ----------
    package : str
        The importable / pip name of the optional package, e.g. ``"textblob"``.
    extra : str
        The name of the ``giskard-checks`` pip extra that installs ``package``
        (e.g. ``"nlp"``). Surfaced verbatim in the error message.
    feature : str, optional
        Human-readable name of the feature that needs ``package``. Defaults
        to ``package`` itself when omitted.

    Returns
    -------
    ModuleType
        The imported module, ready to use.

    Raises
    ------
    OptionalDependencyError
        If ``package`` cannot be imported.
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        feature_name = feature or package
        raise OptionalDependencyError(
            f"The '{package}' package is required for {feature_name} but is not "
            f"installed. Install it with: pip install 'giskard-checks[{extra}]'"
        ) from exc
