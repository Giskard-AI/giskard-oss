"""Privacy-boundary helpers for scan analytics."""

from typing import Any


def safe_target_mode(value: Any) -> str:
    """Return only a fixed target-mode label."""
    return (
        value
        if type(value) is str and value in ("singleturn", "multiturn")
        else "unknown"
    )


def safe_bool(value: Any) -> bool | None:
    """Return booleans without coercing arbitrary caller-owned objects."""
    return value if type(value) is bool else None


def scenario_budget(value: Any) -> str:
    """Bucket a requested scenario limit so it cannot carry identifying numbers."""
    if value is None:
        return "default"
    if type(value) is not int or value < 0:
        return "unknown"
    if value == 0:
        return "none"
    if value <= 10:
        return "small"
    if value <= 100:
        return "medium"
    return "large"
