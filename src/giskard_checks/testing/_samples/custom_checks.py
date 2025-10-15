from __future__ import annotations

from typing import Any

from giskard_checks.core import Check, CheckResult
from giskard_checks.core.interactions import Interaction


@Check.register("starts_with")
class StartsWithCheck(Check):
    """Check that validates if the input starts with a specified prefix.

    This is a sample custom check implementation that demonstrates how to create
    a simple check that validates string input against a prefix pattern.

    Attributes
    ----------
    prefix : str
        The prefix string that the input should start with
    """

    prefix: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:  # type: ignore[override]
        ok = interaction.inputs is not None and str(interaction.inputs).startswith(
            self.prefix
        )
        if ok:
            return CheckResult.success(
                message=f"input starts with '{self.prefix}'",
            )
        return CheckResult.failure(
            message=f"input does not start with '{self.prefix}'",
        )


@Check.register("equals_out")
class EqualsOutputCheck(Check):
    """Check that validates if the output equals an expected value.

    This is a sample custom check implementation that demonstrates how to create
    a simple check that validates the output against an expected string value.

    Attributes
    ----------
    expected : str
        The expected output value to compare against
    """

    expected: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:  # type: ignore[override]
        if interaction.outputs == self.expected:
            return CheckResult.success(
                message="output matched",
            )
        return CheckResult.failure(
            message=f"expected '{self.expected}', got '{interaction.outputs}'",
        )
