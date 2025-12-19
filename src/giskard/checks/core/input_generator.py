from collections.abc import AsyncGenerator

from giskard.core import Discriminated, discriminated_base

from .trace import Trace


@discriminated_base
class InputGenerator[InputType, TraceType: Trace](Discriminated):  # pyright: ignore[reportMissingTypeArgument]
    def __call__(self, trace: TraceType) -> AsyncGenerator[InputType, TraceType]:
        raise NotImplementedError
