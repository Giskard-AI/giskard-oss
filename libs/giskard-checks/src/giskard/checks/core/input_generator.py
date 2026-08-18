from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar, overload

from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .interaction import Trace


@discriminated_base
class InputGenerator[TraceType: "Trace"](Discriminated):  # pyright: ignore[reportMissingTypeArgument]
    """Base class for input generators.

    Subclasses should be registered using the @InputGenerator.register("kind")
    decorator to enable polymorphic serialization and deserialization.

    Unknown fields are rejected (``extra="forbid"``), matching ``Check``.
    Input generators are nested inside ``Interact.inputs`` in persisted
    scenario JSON, so without this pydantic silently drops unrecognized keys:
    a scenario writing ``max_step`` instead of ``max_steps`` would silently
    fall back to the default turn limit instead of the configured one.
    """

    # Rationale and the subclass rule: see ``Discriminated`` in giskard-core.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

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
