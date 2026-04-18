from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Generic

from giskard.core import Discriminated, discriminated_base

if TYPE_CHECKING:
    from .interaction import Trace

from .typevars import InputType, TraceType


@discriminated_base
class InputGenerator(Discriminated, Generic[InputType, TraceType]):
    def __call__(self, trace: TraceType) -> AsyncGenerator[InputType, TraceType]:
        raise NotImplementedError
