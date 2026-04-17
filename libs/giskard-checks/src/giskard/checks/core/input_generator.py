from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Generic

from giskard.core import Discriminated, discriminated_base
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from .interaction import Trace

InputType = TypeVar("InputType")
TraceType = TypeVar(
    "TraceType",
    bound="Trace[Any, Any]",
    default="Trace[Any, Any]",
)


@discriminated_base
class InputGenerator(Discriminated, Generic[InputType, TraceType]):
    def __call__(self, trace: TraceType) -> AsyncGenerator[InputType, TraceType]:
        raise NotImplementedError
