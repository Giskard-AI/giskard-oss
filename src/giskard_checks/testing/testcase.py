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

        The interaction is annotated with its fully-qualified class path to
        allow generic deserialization. Checks include their computed `kind` so
        they can be reconstructed via the global registry.
        """
        interaction_obj = self.interaction
        interaction_cls = interaction_obj.__class__
        interaction_payload = {
            "__type__": f"{interaction_cls.__module__}.{interaction_cls.__name__}",
            "data": interaction_obj.model_dump(),
        }

        checks_payload = [chk.model_dump() for chk in self.checks]

        # Include class paths for checks to enable lazy import during deserialization
        for i, chk in enumerate(self.checks):
            checks_payload[i]["__type__"] = (
                f"{chk.__class__.__module__}.{chk.__class__.__name__}"
            )

        return {
            "name": self.name,
            "interaction": interaction_payload,
            "checks": checks_payload,
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> "TestCase[Any]":
        """Reconstruct a TestCase from a dict produced by `serialize()`.

        Check classes can be lazily imported if the serialized payload contains
        a `__type__` field for each check with the fully-qualified class path.
        """
        from importlib import import_module

        from giskard_checks.core.check import Check

        name = payload.get("name")

        inter_info = payload.get("interaction")
        if not isinstance(inter_info, dict) or "__type__" not in inter_info:
            raise ValueError("Invalid interaction serialization format")
        type_path: str = inter_info["__type__"]
        if "." not in type_path:
            raise ValueError("Invalid interaction type path")
        module_name, class_name = type_path.rsplit(".", 1)
        mod = import_module(module_name)
        inter_cls = getattr(mod, class_name)

        if not issubclass(inter_cls, Interaction):
            raise ValueError("Invalid interaction type")

        interaction = inter_cls.model_validate(inter_info.get("data", {}))

        checks_data = payload.get("checks")
        if not isinstance(checks_data, list):
            raise ValueError("Invalid checks serialization format")
        checks = [Check.from_dict(cd) for cd in checks_data]

        return cls(name=name, interaction=interaction, checks=checks)  # pyright: ignore
