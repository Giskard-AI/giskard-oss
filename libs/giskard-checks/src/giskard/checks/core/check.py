from giskard.core import Discriminated, discriminated_base
from pydantic import Field

from .interaction import Trace
from .result import CheckResult


@discriminated_base
class Check[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Discriminated
):
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

    def __and__(self, other: "Check") -> "Check":
        """Combine two checks with AND semantics (all must pass).

        Example
        -------
        >>> combined = check_a & check_b
        """
        from ..builtin.composition import AllOf

        my_checks: list[Check] = (
            list(self.checks) if hasattr(self, "checks") and self.__class__.__name__ == "AllOf" else [self]
        )
        other_checks: list[Check] = (
            list(other.checks) if hasattr(other, "checks") and other.__class__.__name__ == "AllOf" else [other]
        )
        return AllOf(checks=my_checks + other_checks)

    def __or__(self, other: "Check") -> "Check":
        """Combine two checks with OR semantics (at least one must pass).

        Example
        -------
        >>> combined = check_a | check_b
        """
        from ..builtin.composition import AnyOf

        my_checks: list[Check] = (
            list(self.checks) if hasattr(self, "checks") and self.__class__.__name__ == "AnyOf" else [self]
        )
        other_checks: list[Check] = (
            list(other.checks) if hasattr(other, "checks") and other.__class__.__name__ == "AnyOf" else [other]
        )
        return AnyOf(checks=my_checks + other_checks)

    def __invert__(self) -> "Check":
        """Invert the check result (pass becomes fail, fail becomes pass).

        Example
        -------
        >>> inverted = ~check_a
        """
        from ..builtin.composition import Not

        return Not(check=self)
