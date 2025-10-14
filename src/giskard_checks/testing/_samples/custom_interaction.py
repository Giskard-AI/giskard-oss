from __future__ import annotations

from typing import ClassVar

from giskard_checks.core.interactions import Interaction


@Interaction.register("custom")
class CustomInteraction(Interaction[str, str]):
    """Custom interaction defined in a separate module from checks."""
