from __future__ import annotations

from giskard_checks.core.interactions import Interaction


class CustomInteraction(Interaction[str, str]):
    """Custom interaction defined in a separate module from checks."""
