from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from giskard_checks.checks.extraction_check import ExtractionCheck
from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.interactions import Interaction


@Check.register("equality")
class EqualityCheck(ExtractionCheck):
    """Check that validates if extracted values equal an expected value.

    This check extracts values from an interaction and compares them against a
    specified expected value. It supports both 'any' and 'all' evaluation modes
    for handling multiple extracted values.

    Attributes
    ----------
    expected : Any
        The expected value to compare against the extracted values
    """

    expected: Any = Field(..., description="The expected value to compare against")

    def _evaluate_value(self, value: Any) -> bool:
        """Check if the value equals the expected value."""
        return value == self.expected

    def _create_success_result(self, values: list[Any]) -> CheckResult:
        """Create a success result for equality check."""
        return CheckResult.success(
            message=f"Extracted values match expected value: {self.expected}",
            details={
                "matched_values": values,
                "expected": self.expected,
                "key": self.key,
                "evaluation_mode": self.evaluation_mode,
            },
        )

    def _create_failure_result(self, values: list[Any]) -> CheckResult:
        """Create a failure result for equality check."""
        return CheckResult.failure(
            message=f"Extracted values do not match expected value: {self.expected}",
            details={
                "actual_values": values,
                "expected": self.expected,
                "key": self.key,
                "evaluation_mode": self.evaluation_mode,
            },
        )
