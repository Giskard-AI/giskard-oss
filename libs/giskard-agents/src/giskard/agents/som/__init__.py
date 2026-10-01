"""System One Models for probability prediction."""

from .base import BaseSOM, SOMResponse
from .typesafe import TypeSafeSOM


def resolve_som(model: str) -> BaseSOM | None:
    """Resolve a supported ``provider/model`` name, or return ``None``.

    Unknown providers return ``None`` so callers can use their ordinary LLM
    routing. A supported provider with a missing model raises ``ValueError``.
    """
    provider, _, name = model.partition("/")
    if provider not in BaseSOM.kinds():
        return None
    if not name.strip():
        raise ValueError("Specify a SOM model as 'provider/model'.")
    return BaseSOM.model_validate({"kind": provider, "model": name})


__all__ = [
    "BaseSOM",
    "SOMResponse",
    "TypeSafeSOM",
    "resolve_som",
]
