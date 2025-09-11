from __future__ import annotations

from typing import Any, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.checks.value_based import ExtractionCheck
from giskard_checks.core.check import CheckResult
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class StringMatchingCheck(ExtractionCheck[InteractionT]):
    """Check that validates if a specific string content is present in interaction outputs.

    This check extracts values from an interaction (typically the output field) and
    verifies that the specified content string is contained within the extracted values.
    It supports 'any', 'all', and 'none' evaluation modes for handling multiple extracted values.

    Attributes
    ----------
    content : str
        The string content to search for in the extracted values
    """

    KIND: ClassVar[str | None] = "string_matching"

    content: str = Field(..., description="The string to match in the output")

    def _evaluate_values(self, values: list[Any]) -> bool:
        """Check if the content string is contained in any of the values."""
        # Convert values to strings for comparison
        texts: list[str] = []
        for item in values:
            value = getattr(item, "value", item)
            texts.append(value if isinstance(value, str) else str(value))

        return any(self.content in text for text in texts)

    def _create_success_result(self, values: list[Any]) -> CheckResult:
        """Create a success result for string matching check."""
        return CheckResult.success(message=f"String matching succeeded", details={})

    def _create_failure_result(self, values: list[Any]) -> CheckResult:
        """Create a failure result for string matching check."""
        return CheckResult.failure(message=f"String matching failed", details={})
