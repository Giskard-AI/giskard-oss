from __future__ import annotations

from typing import ClassVar

from giskard_checks.core.interactions import Interaction


class CustomInteraction(Interaction[str, str]):
    """Custom interaction defined in a separate module from checks."""

    KIND: ClassVar[str | None] = "custom"
