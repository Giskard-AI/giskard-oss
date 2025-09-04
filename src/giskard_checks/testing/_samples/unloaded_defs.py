from __future__ import annotations

from giskard_checks.core import Check, CheckResult, CheckSeverity
from giskard_checks.core.interactions import Interaction


class CustomInteraction(Interaction[str, str]):
    """Simple custom interaction to test import-based deserialization."""


class StartsWithCheck(Check[CustomInteraction]):
    KIND = "starts_with"

    prefix: str

    async def run(self, interaction: CustomInteraction) -> CheckResult:  # type: ignore[override]
        ok = interaction.input is not None and str(interaction.input).startswith(
            self.prefix
        )
        if ok:
            return CheckResult.success(
                kind=self.kind,
                name=self.name,
                message=f"input starts with '{self.prefix}'",
                severity=CheckSeverity.INFO,
            )
        return CheckResult.failure(
            kind=self.kind,
            name=self.name,
            message=f"input does not start with '{self.prefix}'",
            severity=CheckSeverity.ERROR,
        )


class EqualsOutputCheck(Check[CustomInteraction]):
    KIND = "equals_out"

    expected: str

    async def run(self, interaction: CustomInteraction) -> CheckResult:  # type: ignore[override]
        if interaction.output == self.expected:
            return CheckResult.success(
                kind=self.kind,
                name=self.name,
                message="output matched",
                severity=CheckSeverity.INFO,
            )
        return CheckResult.failure(
            kind=self.kind,
            name=self.name,
            message=f"expected '{self.expected}', got '{interaction.output}'",
            severity=CheckSeverity.ERROR,
        )
