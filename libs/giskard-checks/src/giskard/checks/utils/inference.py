import inspect
from typing import Any, get_type_hints

from pydantic import PydanticUserError, TypeAdapter

from ..core.interaction.trace import Trace


def _get_param_hints(target: object) -> dict[str, Any]:
    """Return ordered parameter type hints for a callable, excluding 'return'.

    Handles the Python 3.14+ regression where get_type_hints on a callable
    instance returns {} — falls back to inspecting type(target).__call__ directly.
    Returns an empty dict for non-callables or on any resolution error.
    """
    if not callable(target):
        return {}
    try:
        hints = get_type_hints(target)
    except TypeError:
        hints = {}
    except Exception:
        return {}
    param_hints = {k: v for k, v in hints.items() if k != "return"}
    if (
        not param_hints
        and not inspect.isfunction(target)
        and not inspect.ismethod(target)
        and not inspect.isclass(target)
    ):
        try:
            call_hints = get_type_hints(type(target).__call__)
            call_hints.pop("self", None)
            param_hints = {k: v for k, v in call_hints.items() if k != "return"}
        except Exception:
            return {}
    return param_hints


def _infer_input_type(outputs: object) -> type | None:
    """Infer the input type from the first parameter annotation of a callable.

    Returns any pydantic-compatible type, including ``str``. Returns ``None``
    for non-callables, callables with no annotation, and callables whose hints
    cannot be resolved (e.g. forward references to undefined names) or whose
    type is not supported by Pydantic.
    """
    param_hints = _get_param_hints(outputs)
    if not param_hints:
        return None
    first_param_type = next(iter(param_hints.values()))
    try:
        TypeAdapter(first_param_type)
    except (PydanticUserError, TypeError):
        return None
    return first_param_type


def _infer_trace_type(target: object) -> type[Trace] | None:  # pyright: ignore[reportMissingTypeArgument]
    """Infer the trace type from the second parameter annotation of a callable.

    Returns the second parameter type if it is a subclass of ``Trace``,
    otherwise ``None``. Never raises.
    """
    param_hints = _get_param_hints(target)
    if len(param_hints) < 2:
        return None
    second_type = list(param_hints.values())[1]
    try:
        if isinstance(second_type, type) and issubclass(second_type, Trace):
            return second_type
    except TypeError:
        pass
    return None
