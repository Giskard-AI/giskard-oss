from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from giskard.core import Discriminated, discriminated_base

if TYPE_CHECKING:
    from .interaction import Trace


@discriminated_base
class InputGenerator[InputType, TraceType: "Trace"](Discriminated):  # pyright: ignore[reportMissingTypeArgument]
    def __call__(self, trace: TraceType) -> AsyncGenerator[InputType, TraceType]:
        raise NotImplementedError
