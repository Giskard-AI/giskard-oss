"""System One Models for probability prediction."""

from .base import BaseSOM, SOMResponse
from .typesafe import TypeSafeSOM

_PROVIDERS: dict[str, type[BaseSOM]] = {"typesafe": TypeSafeSOM}


def resolve_som(model: str) -> BaseSOM | None:
    """Resolve a supported ``provider/model`` name, or return ``None``.

    Unknown providers return ``None`` so callers can use their ordinary LLM
    routing. A supported provider with a missing model raises ``ValueError``.
    """
    provider, _, name = model.partition("/")
    model_class = _PROVIDERS.get(provider)
    if model_class is None:
        return None
    if not name.strip():
        raise ValueError("Specify a SOM model as 'provider/model'.")
    return model_class(model=name)


__all__ = [
    "BaseSOM",
    "SOMResponse",
    "TypeSafeSOM",
    "resolve_som",
]
