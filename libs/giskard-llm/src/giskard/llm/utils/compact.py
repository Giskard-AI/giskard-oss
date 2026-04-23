from typing import Any


def compact(**kwargs: Any) -> dict[str, Any]:
    """Return a dict with all None values removed."""
    return {k: v for k, v in kwargs.items() if v is not None}
