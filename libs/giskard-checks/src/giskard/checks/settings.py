from giskard.agents.embeddings import BaseEmbeddingModel, EmbeddingModel
from giskard.agents.generators import BaseGenerator, Generator

# Global default generator
_default_generator: BaseGenerator | None = None
_default_embedding_model: BaseEmbeddingModel | None = None


def set_default_generator(generator: "BaseGenerator") -> None:
    """Set the default LLM generator for all checks.

    Parameters
    ----------
    generator : BaseGenerator
        The generator to use as default for all LLM checks.
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


def set_default_embedding_model(embedding_model: "BaseEmbeddingModel") -> None:
    """Set the default embedding model for all checks.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        The embedding model to use as default for all embedding checks.
    """
    global _default_embedding_model
    _default_embedding_model = embedding_model


def get_default_embedding_model() -> BaseEmbeddingModel:
    """Get the current default embedding model.

    Returns
    -------
    BaseEmbeddingModel
        The current default embedding model, or a default text-embedding-3-small model
        if none has been set.
    """
    return _default_embedding_model or EmbeddingModel(model="text-embedding-3-small")
