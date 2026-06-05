import importlib
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_import_lock = threading.Lock()

_REFERENCE_PATH: ContextVar[Path | None] = ContextVar(
    "giskard_checks_reference_path", default=None
)


@contextmanager
def python_reference_path(path: Path | None) -> Iterator[None]:
    """Context manager that sets the reference path for Python target resolution.

    Parameters
    ----------
    path : Path | None
        Path to the definition file being loaded. The file's parent directory
        is used as the first search path for resolving ``python:`` references.

    Yields
    ------
    None
    """
    token = _REFERENCE_PATH.set(path)
    try:
        yield
    finally:
        _REFERENCE_PATH.reset(token)


def validation_reference_path(value: Any) -> Path | None:
    """Return the active reference path for use in Pydantic validators.

    Parameters
    ----------
    value : Any
        A ``Path`` to use directly, or any other value which causes the
        function to fall back to the context-variable path.

    Returns
    -------
    Path | None
        The ``Path`` if *value* is already a ``Path``, otherwise the value
        stored in the ``_REFERENCE_PATH`` context variable.
    """
    if isinstance(value, Path):
        return value
    return _REFERENCE_PATH.get()


def _module_search_paths(path: Path | None) -> list[str]:
    if path is None:
        return []

    search_paths = [str(path.parent.resolve()), str(Path.cwd().resolve())]
    unique_paths: list[str] = []
    for candidate in search_paths:
        if candidate not in unique_paths:
            unique_paths.append(candidate)
    return unique_paths


def _module_belongs_to_search_path(module: Any, search_path: str) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False

    return Path(module_file).resolve().is_relative_to(Path(search_path))


def _module_exists_in_search_path(module_name: str, search_path: str) -> bool:
    module_parts = module_name.split(".")
    module_root = Path(search_path).resolve().joinpath(*module_parts)
    return (
        module_root.with_suffix(".py").exists()
        or (module_root / "__init__.py").exists()
    )


def resolve_python_reference(value: str, *, path: Path | None = None) -> Any:
    """Resolve a ``python:module.symbol`` reference to a callable.

    Parameters
    ----------
    value : str
        A reference string of the form ``python:module.attribute``.
    path : Path | None, optional
        Path to the definition file. The file's parent directory is prepended
        to ``sys.path`` so that modules local to the definition can be found.

    Returns
    -------
    Any
        The resolved attribute from the imported module.

    Raises
    ------
    ValueError
        If *value* does not start with ``python:``, if the format is invalid,
        if the module cannot be imported, or if the attribute does not exist.
    """
    if not value.startswith("python:"):
        raise ValueError(
            f"Unsupported Python reference '{value}'. Expected 'python:module.symbol'."
        )

    import_path = value.removeprefix("python:")
    module_name, _, attribute_path = import_path.rpartition(".")
    if not module_name or not attribute_path:
        raise ValueError(
            f"Invalid Python reference '{value}'. Expected 'python:module.symbol'."
        )

    search_paths = _module_search_paths(path)
    with _import_lock:
        original_sys_path = sys.path[:]
        try:
            for search_path in reversed(search_paths):
                if search_path not in sys.path:
                    sys.path.insert(0, search_path)

            cached_module = sys.modules.get(module_name)
            if cached_module is not None and search_paths:
                preferred_search_path = search_paths[0]
                if _module_exists_in_search_path(module_name, preferred_search_path):
                    if not _module_belongs_to_search_path(
                        cached_module, preferred_search_path
                    ):
                        sys.modules.pop(module_name, None)

            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Could not import module '{module_name}' for Python reference '{value}': {exc}"
            ) from exc
        finally:
            sys.path[:] = original_sys_path

    try:
        return getattr(module, attribute_path)
    except AttributeError as exc:
        raise ValueError(
            f"Could not resolve attribute '{attribute_path}' for Python reference '{value}'."
        ) from exc
