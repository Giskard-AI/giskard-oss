from typing import TYPE_CHECKING, ClassVar

from giskard.core import Discriminated, discriminated_base
from pydantic import ConfigDict, Field

from .interaction import Trace

if TYPE_CHECKING:
    from .result import CheckResult


@discriminated_base
class Check[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Discriminated
):
    """Base class for checks.

    Subclasses should be registered using the @Check.register("kind") decorator
    to enable polymorphic serialization and deserialization.

    Unknown fields are rejected (``extra="forbid"``). Without this, pydantic
    silently drops unrecognized keys: a persisted suite referencing a renamed
    field would fall back to that field's default and the check would run
    green while evaluating the wrong value.

    """

    # Rationale and the subclass rule: see ``Discriminated`` in giskard-core.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, description="Check name")
    description: str | None = Field(default=None, description="Check description")

    async def run(self, trace: TraceType) -> "CheckResult":
        """Execute the check against the provided trace.

        Subclasses must override this method and return a `CheckResult`. The
        implementation may be async.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history. Access the current
            interaction via `trace.last` (preferred in prompt templates) or
            `trace.interactions[-1]` if available.

        Returns
        -------
        CheckResult
            The result of the check evaluation.
        """
        raise NotImplementedError
