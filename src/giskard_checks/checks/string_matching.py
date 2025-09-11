from __future__ import annotations

from typing import Any, ClassVar, TypeVar

from pydantic import Field

from giskard_checks.core.check import Check, CheckResult
from giskard_checks.core.extraction import Extractor, JsonPathExtractor
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class StringMatchingCheck(Check[InteractionT]):
    KIND: ClassVar[str | None] = "string_matching"

    content: str = Field(..., description="The string to match in the output")

    # Optional new extractor strategy (preferred)
    extractor: Extractor | None = Field(
        default=None, description="Optional extractor for selecting values"
    )
    # Backcompat convenience to select values via JSONPath when no extractor is provided
    key: str | None = Field(default=None, description="JSON path to the key to check")
    match_all: bool = Field(
        default=False, description="Whether all strings should match"
    )

    async def run(self, interaction: InteractionT) -> CheckResult:
        # Extract values using configured extractor or fall back to JSONPath/raw output
        if self.extractor is not None:
            values = self.extractor.extract(interaction)
        elif self.key:
            values = JsonPathExtractor(key=self.key).extract(interaction)
        else:
            # Default to selecting the output field via JSONPath
            values = JsonPathExtractor(key="output").extract(interaction)

        texts: list[str] = []
        for item in values:
            value = getattr(item, "value", item)
            texts.append(value if isinstance(value, str) else str(value))

        if self.match_all:
            matched = all(self.content in text for text in texts)
        else:
            matched = any(self.content in text for text in texts)

        if matched:
            return CheckResult.success(message=f"String matching succeeded", details={})
        else:
            return CheckResult.failure(message=f"String matching failed", details={})
