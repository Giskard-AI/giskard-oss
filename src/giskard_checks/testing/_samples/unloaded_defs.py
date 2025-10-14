from __future__ import annotations

from typing import Any

from giskard_checks.core import Check, CheckResult
from giskard_checks.core.interactions import Interaction


class StartsWithCheck(Check):
    KIND = "starts_with"

    prefix: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:  # type: ignore[override]
        ok = interaction.input is not None and str(interaction.input).startswith(
            self.prefix
        )
        if ok:
            return CheckResult.success(
                message=f"input starts with '{self.prefix}'",
            )
        return CheckResult.failure(
            message=f"input does not start with '{self.prefix}'",
        )


class EqualsOutputCheck(Check):
    KIND = "equals_out"

    expected: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:  # type: ignore[override]
        if interaction.output == self.expected:
            return CheckResult.success(
                message="output matched",
            )
        return CheckResult.failure(
            message=f"expected '{self.expected}', got '{interaction.output}'",
        )
