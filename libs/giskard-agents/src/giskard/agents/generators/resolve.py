"""Factory helpers for picking a generator backend from a model string."""

from giskard.llm import supports_native

from .base import BaseGenerator
from .giskard_llm_generator import GiskardLLMGenerator


def resolve_generator(model: str) -> BaseGenerator:
    """Return the best generator backend for *model*.

    Uses ``GiskardLLMGenerator`` when giskard-llm supports the provider natively;
    otherwise falls back to ``LiteLLMGenerator``.

    Parameters
    ----------
    model
        Model identifier (e.g. ``openai/gpt-4o-mini`` or ``deepseek/deepseek-chat``).

    Returns
    -------
    BaseGenerator
        A generator configured for *model*.

    Raises
    ------
    ImportError
        When the native path is unavailable and the optional ``litellm`` extra
        is not installed.
    """
    if supports_native(model, "completion"):
        return GiskardLLMGenerator(model=model)

    from .litellm_generator import LiteLLMGenerator

    return LiteLLMGenerator(model=model)
