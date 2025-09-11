from __future__ import annotations

from typing import Any, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.checks.value_based import ExtractionCheck
from giskard_checks.core.check import CheckResult
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class StringMatchingCheck(ExtractionCheck[InteractionT]):
    KIND: ClassVar[str | None] = "string_matching"

    content: str = Field(..., description="The string to match in the output")
    # Backward compatibility - maps to evaluation_mode
    match_all: bool = Field(
        default=False,
        description="Whether all strings should match (maps to evaluation_mode='all')",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Map match_all to evaluation_mode for backward compatibility
        if "match_all" in data:
            self.evaluation_mode = "all" if data["match_all"] else "any"

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
