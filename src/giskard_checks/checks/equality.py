from __future__ import annotations

from typing import Any, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.checks.value_based import ExtractionCheck
from giskard_checks.core.check import CheckResult
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class EqualityCheck(ExtractionCheck[InteractionT]):
    KIND: ClassVar[str | None] = "equality"

    expected: Any = Field(..., description="The expected value to compare against")

    def _evaluate_values(self, values: list[Any]) -> bool:
        """Check if any of the values equals the expected value."""
        return any(value == self.expected for value in values)

    def _create_success_result(self, values: list[Any]) -> CheckResult:
        """Create a success result for equality check."""
        return CheckResult.success(
            message=f"Input is equal to {self.expected}", details={}
        )

    def _create_failure_result(self, values: list[Any]) -> CheckResult:
        """Create a failure result for equality check."""
        return CheckResult.failure(
            message=f"Input is not equal to {self.expected}",
            details={
                "actual": values,
                "expected": self.expected,
                "key": self.key,
            },
        )
