from typing import Any

from typing_extensions import TypeVar

from .interaction.trace import Trace

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
TraceType = TypeVar(
    "TraceType",
    bound=Trace[Any, Any],
    default=Trace[InputType, OutputType],
)
