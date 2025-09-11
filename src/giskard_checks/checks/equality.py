from __future__ import annotations

from typing import Any, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.extraction import Extractor, JsonPathExtractor
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class EqualityCheck(Check[InteractionT]):
    KIND = "equality"

    expected: Any
    # Optional new extractor strategy (preferred)
    extractor: Extractor | None = Field(
        default=None, description="Optional extractor for selecting values"
    )
    # Backcompat convenience to select values via JSONPath when no extractor is provided
    key: str | None = Field(default=None, description="JSON path to the key to check")

    async def run(self, interaction: InteractionT) -> CheckResult:
        # Determine values to compare
        if self.extractor is not None:
            values = self.extractor.extract(interaction)
        elif self.key:
            values = JsonPathExtractor(key=self.key).extract(interaction)
        else:
            # Default to selecting the output field via JSONPath
            values = JsonPathExtractor(key="output").extract(interaction)

        matched = any(value == self.expected for value in values)
        if matched:
            return CheckResult.success(
                message=f"Input is equal to {self.expected}", details={}
            )

        return CheckResult.failure(
            message=f"Input is not equal to {self.expected}",
            details={
                "actual": values,
                "expected": self.expected,
                "key": self.key,
            },
        )
