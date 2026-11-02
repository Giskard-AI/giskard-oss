import inspect
from itertools import islice
from typing import Any, get_origin, get_type_hints

from pydantic import PydanticUserError, TypeAdapter

from ..core.interaction.trace import Trace
from ..core.types import Target


def _get_param_hints(target: object) -> dict[str, Any]:
    """Return parameter type hints in signature order, excluding 'return'.

    Unannotated parameters map to ``None`` (``get_type_hints`` omits them), so
    positions match the signature. Falls back to type(target).__call__ for
    Python 3.14+ callable-instance regression.
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
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return param_hints
    return {name: param_hints.get(name) for name in parameters}


def _infer_input_type(outputs: object) -> type | None:
    """Return first parameter's pydantic-compatible type, or None."""
    first_param_type = next(iter(_get_param_hints(outputs).values()), None)
    if first_param_type is None:
        return None
    try:
        TypeAdapter(first_param_type)
    except (PydanticUserError, TypeError):
        return None
    return first_param_type


def _infer_trace_type[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
) -> type[TraceType] | None:
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
