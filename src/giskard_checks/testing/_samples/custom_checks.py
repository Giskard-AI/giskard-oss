from __future__ import annotations

from giskard_checks.core import Check, CheckResult
from giskard_checks.testing._samples.custom_interaction import CustomInteraction


class StartsWithCheck(Check[CustomInteraction]):
    """Check that validates if the input starts with a specified prefix.

    This is a sample custom check implementation that demonstrates how to create
    a simple check that validates string input against a prefix pattern.

    Attributes
    ----------
    prefix : str
        The prefix string that the input should start with
    """

    KIND = "starts_with"

    prefix: str

    async def run(self, interaction: CustomInteraction) -> CheckResult:  # type: ignore[override]
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


class EqualsOutputCheck(Check[CustomInteraction]):
    """Check that validates if the output equals an expected value.

    This is a sample custom check implementation that demonstrates how to create
    a simple check that validates the output against an expected string value.

    Attributes
    ----------
    expected : str
        The expected output value to compare against
    """

    KIND = "equals_out"

    expected: str

    async def run(self, interaction: CustomInteraction) -> CheckResult:  # type: ignore[override]
        if interaction.output == self.expected:
            return CheckResult.success(
                message="output matched",
            )
        return CheckResult.failure(
            message=f"expected '{self.expected}', got '{interaction.output}'",
        )
