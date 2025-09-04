from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, SkipValidation, field_validator

from giskard_checks.core.check import Check
from giskard_checks.core.interactions import Interaction

InteractionT = TypeVar("InteractionT", bound=Interaction[Any, Any])


class TestCase(BaseModel, Generic[InteractionT]):
    name: str | None = Field(None, description="Test case name")
    interaction: InteractionT = Field(..., description="Test case interaction")
    checks: Sequence[Check] = Field(..., description="Test case checks")  # pyright: ignore

    async def run(self):
        # Lazy import to avoid circular dependency with runner importing TestCase
        from giskard_checks.testing.runner import get_runner

        runner = get_runner()
        return await runner.run(self)
