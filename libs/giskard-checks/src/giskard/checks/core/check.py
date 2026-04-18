from typing import Generic

from giskard.core import Discriminated, discriminated_base
from pydantic import Field

from .result import CheckResult
from .typevars import InputType, OutputType, TraceType


@discriminated_base
class Check(Discriminated, Generic[InputType, OutputType, TraceType]):
    """Base class for checks.

    Subclasses should be registered using the @Check.register("kind") decorator
    to enable polymorphic serialization and deserialization.
    """

    name: str | None = Field(default=None, description="Check name")
    description: str | None = Field(default=None, description="Check description")

    async def run(self, trace: TraceType) -> CheckResult:
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
