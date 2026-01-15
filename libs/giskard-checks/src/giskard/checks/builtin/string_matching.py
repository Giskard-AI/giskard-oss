from __future__ import annotations

from typing import Any, override

from pydantic import Field

from ..core.check import Check
from ..core.result import CheckResult
from ..core.trace import Trace
from .extraction_check import ExtractionCheck


@Check.register("string_matching")
class StringMatching[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    ExtractionCheck[InputType, OutputType, TraceType]
):
    """Check that validates if a specific string content is present in trace outputs.

    This check extracts values from a trace (typically the output field) and
    verifies that the specified content string is contained within the extracted values.
    It supports 'any', 'all', and 'none' evaluation modes for handling multiple extracted values.

    Attributes
    ----------
    content : str
        The string content to search for in the extracted values
    """

    content: str = Field(..., description="The string to match in the output")

    @override
    def _evaluate_value(self, value: Any) -> bool:
        """Check if the content string is contained in the value."""
        # Convert value to string for comparison
        item_value = getattr(value, "value", value)
        text = item_value if isinstance(item_value, str) else str(item_value)

        return self.content in text

    @override
    def _create_success_result(self, values: list[Any]) -> CheckResult:
        """Create a success result for string matching check."""
        return CheckResult.success(
            message=f"String '{self.content}' found in extracted values",
            details={
                "matched_values": values,
                "content": self.content,
                "key": self.key,
                "evaluation_mode": self.evaluation_mode,
            },
        )

    @override
    def _create_failure_result(self, values: list[Any]) -> CheckResult:
        """Create a failure result for string matching check."""
        return CheckResult.failure(
            message=f"String '{self.content}' not found in extracted values",
            details={
                "searched_values": values,
                "content": self.content,
                "key": self.key,
                "evaluation_mode": self.evaluation_mode,
            },
        )
