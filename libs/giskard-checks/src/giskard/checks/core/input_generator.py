from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, overload

from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel

if TYPE_CHECKING:
    from .interaction import Trace


@discriminated_base
class InputGenerator[TraceType: "Trace"](Discriminated):  # pyright: ignore[reportMissingTypeArgument]
    @overload
    def __call__(
        self, trace: TraceType, input_type: type[str] | None = None
    ) -> AsyncGenerator[str, TraceType]: ...
    @overload
    def __call__[T: BaseModel](
        self, trace: TraceType, input_type: type[T]
    ) -> AsyncGenerator[T, TraceType]: ...
    @overload
    def __call__[T](
        self, trace: TraceType, input_type: type[T]
    ) -> AsyncGenerator[T, TraceType]: ...
    def __call__(
        self, trace: TraceType, input_type: type[Any] | None = None
    ) -> AsyncGenerator[Any, TraceType]:
        raise NotImplementedError
