"""Helpers for checks that depend on optional third-party packages."""

import importlib
from typing import Any


def require_optional_dependency(module_name: str, *, install_hint: str) -> Any:
    """Import ``module_name`` or raise ``ValueError`` with ``install_hint``.

    Parameters
    ----------
    module_name : str
        Module to import (for example ``"textstat"`` or ``"regorus"``).
    install_hint : str
        Human-readable install instructions included in the validation error.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    ValueError
        If the module cannot be imported.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as err:
        raise ValueError(install_hint) from err
