from counterpoint.generators import Generator
from counterpoint.generators.base import BaseGenerator

# Global default generator
_default_generator: BaseGenerator | None = None


def set_default_generator(generator: "BaseGenerator") -> None:
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator
        The counterpoint generator to use as default for all LLM checks.
    """
    global _default_generator
    _default_generator = generator


def get_default_generator() -> BaseGenerator:
    """Get the current default generator.

    Returns
    -------
    BaseGenerator
        The current default generator, or a default GPT-4o-mini generator
        if none has been set.
    """
    return _default_generator or Generator(model="openai/gpt-4o-mini")
