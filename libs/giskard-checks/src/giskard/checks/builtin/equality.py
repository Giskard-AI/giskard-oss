from __future__ import annotations

from pydantic import Field

from ..core.check import Check
from ..core.extraction import provided_or_resolve
from ..core.result import CheckResult
from ..core.trace import Trace


@Check.register("equality")
class Equality[InputType, OutputType, TraceType: Trace, ExpectedType](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates if extracted values equal an expected value.

    This check extracts values from a trace and compares them against a
    specified expected value. It supports both 'any' and 'all' evaluation modes
    for handling multiple extracted values.

    .. warning::
        For object instances, this check uses Python's ``__eq__`` method for
        comparison. The behavior depends on how the object's ``__eq__`` method
        is implemented. For custom objects, ensure that ``__eq__`` is properly
        defined to match your comparison requirements.

    Attributes
    ----------
    expected_value : ExpectedType
        The expected value to compare against the extracted values
    actual_value_key : str
        The key to extract the actual value from the trace
    """

    actual_value_key: str = Field(
        ..., description="The key to extract the actual value from the trace"
    )
    expected_value: ExpectedType = Field(
        ...,
        description="The expected value to compare against. Use Python's ``__eq__`` method for instance comparison.",
    )

    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the check against the provided trace.

        Parameters
        ----------
        trace : TraceType
            The trace containing interaction history. Access the current
            interaction via `trace.last` (preferred in prompt templates) or
            `trace.interactions[-1]` if available.
        """
        actual_value = provided_or_resolve(None, trace, self.actual_value_key)

        details = {
            "actual_value": actual_value,
            "expected_value": self.expected_value,
        }

        if actual_value is None and self.expected_value is not None:
            return CheckResult.failure(
                message=f"No value found for key '{self.actual_value_key}', expected {repr(self.expected_value)}.",
                details=details,
            )

        if actual_value == self.expected_value:
            return CheckResult.success(
                message="The actual value matches the expected value.",
                details=details,
            )

        return CheckResult.failure(
            message=f"Expected value {repr(self.expected_value)} but got {repr(actual_value)}",
            details=details,
        )
