import inspect
from itertools import islice
from typing import Any, get_origin, get_type_hints

from pydantic import PydanticUserError, TypeAdapter

from ..core.interaction.trace import Trace


def _get_param_hints(target: object) -> dict[str, Any]:
    """Return ordered parameter type hints, excluding 'return'; falls back to type(target).__call__ for Python 3.14+ callable-instance regression."""
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
    """Return first parameter's pydantic-compatible type, or None."""
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
    """Return second parameter's type if it is a Trace subclass, otherwise None."""
    param_hints = _get_param_hints(target)
    if len(param_hints) < 2:
        return None
    second_type = next(islice(param_hints.values(), 1, None))
    try:
        origin = get_origin(second_type) or second_type
        if isinstance(origin, type) and issubclass(origin, Trace):
            return second_type
    except TypeError:
        pass
    return None
