from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field, SkipValidation, field_validator

from giskard_checks.core.check import Check
from giskard_checks.core.interactions import Interaction

if TYPE_CHECKING:
    from giskard_checks.testing.runner import TestCaseResult


"""Test case model and runner integration.

`TestCase` binds a specific `Interaction` instance with a sequence of `Check`s
and delegates execution to a `TestRunner`. It offers a single `run()` method
that returns a `TestCaseResult` summarizing the outcomes.
"""

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class TestCase(BaseModel, Generic[InteractionT]):
    """Bundle a single interaction with a set of checks to execute.

    Attributes
    ----------
    name:
        Optional label for the test case.
    interaction:
        The interaction under test.
    checks:
        Sequence of checks to run against the interaction.
    """

    # Prevent pytest from attempting to collect this class as a test
    __test__ = False

    name: str | None = Field(None, description="Test case name")
    # Validation is skipped for the interaction field to allow for generic deserialization
    interaction: SkipValidation[InteractionT] = Field(
        ..., description="Test case interaction"
    )
    checks: Sequence[SkipValidation[Check[InteractionT]]] = Field(
        ..., description="Test case checks"
    )

    @field_validator("interaction")
    def validate_interaction(cls, val):
        if not isinstance(val, Interaction):
            raise TypeError(
                "Wrong type for 'interaction', must be subclass of Interaction"
            )
        return val

    @field_validator("checks")
    def validate_checks(cls, val):
        for chk in val:
            if not isinstance(chk, Check):
                raise TypeError("Wrong type for 'checks', must be subclass of Check")
        return val

    async def run(self) -> TestCaseResult:
        """Execute the test case using the configured `TestRunner`."""
        # Lazy import to avoid circular dependency with runner importing TestCase
        from giskard_checks.testing.runner import get_runner

        runner = get_runner()
        return await runner.run(self)

    # --- Serialization helpers -------------------------------------------------
    def serialize(self) -> dict[str, Any]:
        """Serialize the TestCase into a JSON-friendly dict.

        Uses the new registry-based serialization for both interactions and checks.
        The `kind` field in each object enables registry-based deserialization.
        """
        interaction_payload = self.interaction.serialize()
        checks_payload = [chk.serialize() for chk in self.checks]

        return {
            "name": self.name,
            "interaction": interaction_payload,
            "checks": checks_payload,
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> "TestCase[Any]":
        """Reconstruct a TestCase from a dict produced by `serialize()`.

        Uses the new registry-based deserialization for both interactions and checks.
        """
        from giskard_checks.core.check import Check

        name = payload.get("name")

        # Deserialize interaction using registry
        inter_info = payload.get("interaction")
        if not isinstance(inter_info, dict):
            raise ValueError("Invalid interaction serialization format")
        interaction = Interaction.deserialize(inter_info)

        # Deserialize checks using registry
        checks_data = payload.get("checks")
        if not isinstance(checks_data, list):
            raise ValueError("Invalid checks serialization format")
        checks = [Check.deserialize(cd) for cd in checks_data]

        return cls(name=name, interaction=interaction, checks=checks)  # pyright: ignore
