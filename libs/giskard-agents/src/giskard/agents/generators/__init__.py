from typing import TYPE_CHECKING, Any

from ._types import GenerationParams
from .base import BaseGenerator
from .giskard_llm_generator import GiskardLLMGenerator, GiskardLLMRetryMiddleware
from .middleware import (
    CompletionMiddleware,
    RateLimiterMiddleware,
    RetryMiddleware,
    RetryPolicy,
)

Generator = GiskardLLMGenerator

if TYPE_CHECKING:
    # Only imported for type checkers; the real import requires the optional
    # 'litellm' extra and happens lazily via ``__getattr__``.
    from .litellm_generator import LiteLLMGenerator, LiteLLMRetryMiddleware

__all__ = [
    "Generator",
    "GenerationParams",
    "BaseGenerator",
    "GiskardLLMGenerator",
    "GiskardLLMRetryMiddleware",
    "LiteLLMGenerator",
    "LiteLLMRetryMiddleware",
    "CompletionMiddleware",
    "RetryMiddleware",
    "RetryPolicy",
    "RateLimiterMiddleware",
]


_LITELLM_ATTRS = {"LiteLLMGenerator", "LiteLLMRetryMiddleware"}


def __getattr__(name: str) -> Any:
    if name in _LITELLM_ATTRS:
        # Triggers ImportError with an install hint if the extra is missing.
        from . import litellm_generator

        return getattr(litellm_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
